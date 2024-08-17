#!/usr/bin/python3
import discord
from discord.ext import commands
import psutil
import distro
import shutil
import cpuinfo
import platform
import nvidia_smi
import ollama
import os
import json
import typing
import functools
import asyncio
import string
from dotenv import load_dotenv
import os
import random
import sys
from typing import Sequence, Mapping, Any, Union
import torch

load_dotenv()
comfyui_dir = os.environ["COMFYUI_DIR"]
sys.path.append(os.environ["COMFYUI_DIR"])
from nodes import (
    DualCLIPLoader,
    CLIPTextEncode,
    UNETLoader,
    VAELoader,
    VAEDecode,
    SaveImage,
    KSampler,
    NODE_CLASS_MAPPINGS,
    EmptyLatentImage,
)

client = ollama.AsyncClient()


def get_value_at_index(obj: Union[Sequence, Mapping], index: int) -> Any:
    """Returns the value at the given index of a sequence or mapping.

    If the object is a sequence (like list or string), returns the value at the given index.
    If the object is a mapping (like a dictionary), returns the value at the index-th key.

    Some return a dictionary, in these cases, we look for the "results" key
    Args:
        obj (Union[Sequence, Mapping]): The object to retrieve the value from.
        index (int): The index of the value to retrieve.

    Returns:
        Any: The value at the given index.

    Raises:
        IndexError: If the index is out of bounds for the object and the object is not a mapping.
    """
    try:
        return obj[index]
    except KeyError:
        return obj["result"][index]


intents = discord.Intents.all()
bot = commands.Bot(intents=intents)
nvidia_smi.nvmlInit()
deviceCount = nvidia_smi.nvmlDeviceGetCount()
chat_hist = {}


def get_gpu(i):
    handle = nvidia_smi.nvmlDeviceGetHandleByIndex(i)
    util = nvidia_smi.nvmlDeviceGetUtilizationRates(handle)
    mem = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
    return [mem.used, mem.total, util.gpu]


def to_thread(func: typing.Callable) -> typing.Coroutine:
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        return await asyncio.to_thread(func, *args, **kwargs)

    return wrapper


def id_generator(size=8, chars=string.ascii_letters + string.digits):
    return "".join(random.choice(chars) for _ in range(size))


@bot.slash_command(description="Chat with a LLM")
async def chat(
    ctx,
    message=discord.Option(str, "Message to send"),
    model=discord.Option(str, "Model to use", default=os.environ["DEFAULT_MODEL"]),
):
    if ctx.author.id not in chat_hist:
        chat_hist.update({ctx.author.id: []})
    chat_hist[ctx.author.id].append(
        {
            "role": "user",
            "content": message,
        }
    )
    await ctx.response.defer()
    response = await client.chat(model=model, messages=chat_hist[ctx.author.id])
    msg = response["message"]
    chat_hist[ctx.author.id].append(msg)
    tmp = msg["content"]
    ff = open(f"history/{ctx.author.id}.json", "w")
    ff.write(json.dumps(chat_hist[ctx.author.id]))
    if len(tmp) <= 4000:
        embed = discord.Embed(title="Response:", color=0x007FFF, description=tmp)
        embed.add_field(
            name="Prompt token count:",
            value=(
                f"{response['prompt_eval_count']} token" + "s"
                if response["prompt_eval_count"] > 1
                else ""
            ),
            inline=False,
        )
        embed.add_field(
            name="Prompt processing speed:",
            value=f"{response['prompt_eval_count']/response['prompt_eval_duration']*1e9:.1f} t/s",
            inline=False,
        )
        embed.add_field(
            name="Generated token count:",
            value=(
                f"{response['eval_count']} token" + "s"
                if response["eval_count"] > 1
                else ""
            ),
            inline=False,
        )
        embed.add_field(
            name="Generation speed:",
            value=f"{response['eval_count']/response['eval_duration']*1e9:.1f} t/s",
            inline=False,
        )
        await ctx.followup.send(embed=embed)
    else:
        f = open("message.txt", "w")
        f.write(tmp)
        f.close()
        await ctx.followup.send(file=discord.File("message.txt"))


@bot.slash_command(description="Clear the chat history")
async def clear(ctx):
    embed = discord.Embed(title="Chat history cleared!", color=0x007FFF)
    if ctx.author.id in chat_hist:
        r = []
        if (
            len(chat_hist[ctx.author.id]) > 0
            and chat_hist[ctx.author.id][0]["role"] == "system"
        ):
            r = [chat_hist[ctx.author.id][0]]
        chat_hist.update({ctx.author.id: r})
    if ctx.author.id in chat_hist:
        ff = open(f"history/{ctx.author.id}.json", "w")
        ff.write(json.dumps(chat_hist[ctx.author.id]))
    await ctx.respond(embed=embed)


@bot.slash_command(description="Set system prompt")
async def system(ctx, system=discord.Option(str, "System prompt", default="")):
    embed = discord.Embed(title="System prompt set!", color=0x007FFF)
    if system == "":
        if ctx.author.id in chat_hist:
            if chat_hist[ctx.author.id][0]["role"] == "system":
                chat_hist[ctx.author.id].pop(0)
    else:
        if ctx.author.id in chat_hist:
            if chat_hist[ctx.author.id][0]["role"] == "system":
                chat_hist[ctx.author.id][0]["content"] = system
            else:
                chat_hist[ctx.author.id].insert(
                    [{"role": "system", "content": system}], 0
                )
        else:
            chat_hist.update({ctx.author.id: [{"role": "system", "content": system}]})
    if ctx.author.id in chat_hist:
        ff = open(f"history/{ctx.author.id}.json", "w")
        ff.write(json.dumps(chat_hist[ctx.author.id]))
    await ctx.respond(embed=embed)


@bot.slash_command(description="Show Ollama models")
async def ps(ctx):
    embed = discord.Embed(title="Ollama ps", color=0x007FFF)
    models = ollama.ps()["models"]
    for i in range(len(models)):
        embed.add_field(name=f"Model {i+1} name: ", value=models[i]["name"])
        embed.add_field(
            name=f"Model {i+1} size: ", value=f"{models[i]['size']/1e9:.1f} GB"
        )
        embed.add_field(
            name=f"Model {i+1} processors: ",
            value=f"{models[i]['size_vram']/models[i]['size']*100:.1f}% GPU ({models[i]['size_vram']/1e9:.1f} GB), {(models[i]['size']-models[i]['size_vram'])/models[i]['size']*100:.1f}% CPU ({(models[i]['size']-models[i]['size_vram'])/1e9:.1f} GB)",
            inline=False,
        )
    await ctx.respond(embed=embed)


@bot.slash_command(description="Show system stats")
async def stats(ctx):
    embed = discord.Embed(title="System Stats", color=0x007FFF)
    cpu_percent = psutil.cpu_percent()
    memory_percent = psutil.virtual_memory().percent
    memory_gb = psutil.virtual_memory().total / 1e9
    memory_gb_used = memory_gb - psutil.virtual_memory().available / 1e9
    starttime = int(psutil.boot_time())
    total, used, free = shutil.disk_usage("/")
    os = distro.name(pretty=True)
    embed.add_field(name="CPU Usage:", value=f"{cpu_percent}%", inline=False)
    embed.add_field(
        name="Memory Usage:",
        value=f"{memory_percent}% ({memory_gb_used:.1f}/{memory_gb:.1f} GB)",
        inline=False,
    )
    if deviceCount:
        if deviceCount == 1:
            gpu_stats = get_gpu(0)
            gpu_percent = gpu_stats[2]
            gpu_memory_used = gpu_stats[0] / 1e9
            gpu_memory_total = gpu_stats[1] / 1e9
            embed.add_field(name="GPU Usage:", value=f"{gpu_percent}%", inline=False)
            embed.add_field(
                name="GPU Memory Usage:",
                value=f"{gpu_memory_used*100/gpu_memory_total:.1f}% ({gpu_memory_used:.1f}/{gpu_memory_total:.1f} GB)",
                inline=False,
            )
        else:
            for i in range(deviceCount):
                gpu_stats = get_gpu(i)
                gpu_percent = gpu_stats[2]
                gpu_memory_used = gpu_stats[0] / 1e9
                gpu_memory_total = gpu_stats[1] / 1e9
                embed.add_field(
                    name=f"GPU {i+1} Usage:", value=f"{gpu_percent}%", inline=False
                )
                embed.add_field(
                    name=f"GPU {i+1} Memory Usage:",
                    value=f"{gpu_memory_used*100/gpu_memory_total:.1f}% ({gpu_memory_used:.1f}/{gpu_memory_total:.1f} GB)",
                    inline=False,
                )
    embed.add_field(
        name="Disk Usage:",
        value=f"{used*100/total:.1f}% ({used/1e9:.1f}/{total/1e9:.1f} GB)",
        inline=False,
    )
    embed.add_field(name="Uptime:", value=f"<t:{starttime}:R>", inline=False)
    embed.add_field(name="OS:", value=f"{os}", inline=False)
    embed.add_field(
        name="Kernel:",
        value=f"{platform.uname().system} {platform.uname().release}",
        inline=False,
    )
    embed.add_field(
        name="Processor:",
        value=f"{psutil.cpu_count(logical=True)} x {cpuinfo.get_cpu_info()['brand_raw']}",
        inline=False,
    )
    embed.add_field(
        name="Latency:",
        value=f"{bot.latency*1e3:.1f}ms",
        inline=False,
    )
    await ctx.respond(embed=embed)


def to_thread(func: typing.Callable) -> typing.Coroutine:
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        return await asyncio.to_thread(func, *args, **kwargs)

    return wrapper


loaded_name = "dev"
with torch.inference_mode():
    dualcliploader = DualCLIPLoader()
    clip_model = dualcliploader.load_clip(
        clip_name1="t5xxl_fp8_e4m3fn.safetensors",
        clip_name2="clip_l.safetensors",
        type="flux",
    )
    unetloader = UNETLoader()
    loaded_model = unetloader.load_unet(
        unet_name="flux1-dev-fp8.safetensors", weight_dtype="fp8_e4m3fn"
    )
    vaeloader = VAELoader()
    ae = vaeloader.load_vae(vae_name="ae.sft")
    emptylatentimage = EmptyLatentImage()
    cliptextencode = CLIPTextEncode()
    ksampler = KSampler()
    vaedecode = VAEDecode()
    saveimage = SaveImage()


@to_thread
def generate_image(
    prompt,
    model,
    steps,
    width,
    height,
):
    id = id_generator()
    with torch.inference_mode():
        if loaded_name != model:
            if model == "schnell":
                loaded_model = unetloader.load_unet(
                    unet_name="flux1-schnell-fp8.safetensors", weight_dtype="fp8_e4m3fn"
                )
            else:
                loaded_model = unetloader.load_unet(
                    unet_name="flux1-dev-fp8.safetensors", weight_dtype="fp8_e4m3fn"
                )
            loaded_name = model
        emptylatentimage_2 = emptylatentimage.generate(
            width=width, height=height, batch_size=1
        )
        cliptextencode_3 = cliptextencode.encode(
            text=prompt, clip=get_value_at_index(clip_model, 0)
        )

        cliptextencode_4 = cliptextencode.encode(
            text="", clip=get_value_at_index(clip_model, 0)
        )

        ksampler_1 = ksampler.sample(
            seed=random.randint(1, 2**64),
            steps=steps if steps else 20 if model == "dev" else 4,
            cfg=1,
            sampler_name="euler",
            scheduler="normal",
            denoise=1,
            model=get_value_at_index(loaded_model, 0),
            positive=get_value_at_index(cliptextencode_3, 0),
            negative=get_value_at_index(cliptextencode_4, 0),
            latent_image=get_value_at_index(emptylatentimage_2, 0),
        )
        vaedecode_5 = vaedecode.decode(
            samples=get_value_at_index(ksampler_1, 0),
            vae=get_value_at_index(ae, 0),
        )
        saveimage.save_images(
            filename_prefix=id, images=get_value_at_index(vaedecode_5, 0)
        )
        return os.path.join(comfyui_dir, f"output/{id}_00001_.png")


@bot.slash_command(description="Show system stats")
async def flux(
    ctx,
    prompt=discord.Option(str),
    model=discord.Option(str, default="dev", choices=["dev", "schnell"]),
    steps=discord.Option(int, default=0),
    width=discord.Option(int, default=os.environ["DEFAULT_SIZE"]),
    height=discord.Option(int, default=os.environ["DEFAULT_SIZE"]),
):
    await ctx.response.defer()
    pth = await generate_image(prompt, model, steps, width, height)
    await ctx.followup.send(
        file=discord.File(
            str(pth),
            filename="image.png",
        )
    )
    os.remove(pth)


from os import listdir
from os.path import isfile, join

for f in listdir("history"):
    if isfile(join("history", f)):
        ff = open(join("history", f), "r")
        try:
            chat_hist[int(f.replace(".json", ""))] = json.load(ff)
        except json.decoder.JSONDecodeError:
            pass
print(ollama.ps())
bot.run(os.getenv("TOKEN"))
