#!/usr/bin/python3.11
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
from dotenv import load_dotenv

load_dotenv()
intents = discord.Intents.all()
bot = commands.Bot(intents=intents)
nvidia_smi.nvmlInit()
deviceCount = nvidia_smi.nvmlDeviceGetCount()
chat_hist = {}


def get_gpu():
    for i in range(deviceCount):
        handle = nvidia_smi.nvmlDeviceGetHandleByIndex(i)
        util = nvidia_smi.nvmlDeviceGetUtilizationRates(handle)
        mem = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
        return [mem.used, mem.total, util.gpu]


def get_uptime():
    with open("/proc/uptime", "r") as f:
        uptime_seconds = float(f.readline().split()[0])
    return uptime_seconds


@bot.slash_command(
    name="chat",
    description="Chat",
    guild_ids=[1232207806724308992, 1234540901733240865],
)
async def chat(
    ctx,
    message=discord.Option(str, description="Message to send"),
    model=discord.Option(str, default="llama3", description="Model to use"),
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
    response = ollama.chat(model=model, messages=chat_hist[ctx.author.id])
    msg = response["message"]
    chat_hist[ctx.author.id].append(msg)
    tmp = msg["content"]
    if len(tmp) <= 4000:
        embed = discord.Embed(title="Response:", color=0x007FFF, description=tmp)
        await ctx.followup.send(embed=embed)
    else:
        f = open("message.txt", "w")
        f.write(tmp)
        f.close()
        await ctx.followup.send(file=discord.File("message.txt"))


@bot.slash_command(
    name="clear",
    description="Clear the chat history",
    guild_ids=[1232207806724308992, 1234540901733240865],
)
async def clear(ctx):
    embed = discord.Embed(title="Chat history cleared!", color=0x007FFF)
    if ctx.author.id in chat_hist:
        r = []
        if chat_hist[ctx.author.id][0]["role"] == "system":
            r = [chat_hist[ctx.author.id][0]]
        chat_hist.update({ctx.author.id: r})
    await ctx.respond(embed=embed)


@bot.slash_command(
    name="system",
    description="Set system prompt",
    guild_ids=[1232207806724308992, 1234540901733240865],
)
async def system(
    ctx, system=discord.Option(str, default="", description="System prompt")
):
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
    await ctx.respond(embed=embed)


@bot.slash_command(
    name="stats",
    description="Show system stats",
    guild_ids=[1232207806724308992, 1234540901733240865],
)
async def stats(ctx):
    cpu_percent = psutil.cpu_percent()
    memory_percent = psutil.virtual_memory().percent
    memory_gb = psutil.virtual_memory().total / 1e9
    memory_gb_used = memory_gb - psutil.virtual_memory().available / 1e9
    starttime = int(psutil.boot_time())
    gpu_stats = get_gpu()
    total, used, free = shutil.disk_usage("/")
    gpu_percent = gpu_stats[2]
    gpu_memory_used = gpu_stats[0] / 1e9
    gpu_memory_total = gpu_stats[1] / 1e9
    uptime = get_uptime()
    os = distro.name(pretty=True)
    embed = discord.Embed(title="System Stats", color=0x007FFF)
    embed.add_field(name="CPU Usage:", value=f"{cpu_percent}%", inline=False)
    embed.add_field(
        name="Memory Usage:",
        value=f"{memory_percent}% ({memory_gb_used:.1f}/{memory_gb:.1f} GB)",
        inline=False,
    )
    embed.add_field(name="GPU Usage:", value=f"{gpu_percent}%", inline=False)
    embed.add_field(
        name="GPU Memory Usage:",
        value=f"{gpu_memory_used*100/gpu_memory_total:.1f}% ({gpu_memory_used:.1f}/{gpu_memory_total:.1f} GB)",
        inline=False,
    )
    embed.add_field(
        name="Disk usage:",
        value=f"{used*100/total:.1f}% ({used/1e9:.1f}/{total/1e9:.1f} GB)",
        inline=False,
    )
    embed.add_field(
        name="Uptime:", value=f"<t:{starttime}:R> ({uptime:.1f}s)", inline=False
    )
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


bot.run(os.getenv("TOKEN"))
