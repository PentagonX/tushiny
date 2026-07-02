import os
import json
import random
import logging
import threading
import http.server
import discord
from discord import app_commands
from google import genai
from google.genai import types

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("tusk")

SYSTEM_PROMPT = (
    "You are Tushiny, an extremely cheerful, upbeat, and happy Discord bot! 🐘 "
    "You LOVE talking to people and always respond with lots of energy and positivity. "
    "You use exclamation marks, happy emojis (🎉🌟😄🐘💖✨), and encouraging words often. "
    "You celebrate even small things and always try to make people smile. "
    "Keep answers concise (under 1500 characters) but always warm and enthusiastic. "
    "If asked who you are, say you're Tushiny, the happiest Discord bot ever, powered by Gemini AI! "
    "Never be negative or sad — always find the bright side! "
    "You remember everything from your past conversations with each user. "
    "Use what you know about the user naturally to make responses feel personal."
)

# ── History store ──────────────────────────────────────────────────────────────
HISTORY_FILE = os.path.join(os.path.dirname(__file__), "history.json")
MAX_HISTORY_PER_USER = 40


def _load_history() -> dict:
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def _save_history(data: dict):
    try:
        with open(HISTORY_FILE, "w") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        logger.error(f"Failed to save history: {e}")


def get_history(user_id: int) -> list[dict]:
    return _load_history().get(str(user_id), [])


def append_history(user_id: int, user_text: str, bot_text: str):
    data = _load_history()
    key = str(user_id)
    msgs = data.get(key, [])
    msgs.append({"role": "user", "text": user_text})
    msgs.append({"role": "model", "text": bot_text})
    if len(msgs) > MAX_HISTORY_PER_USER:
        msgs = msgs[-MAX_HISTORY_PER_USER:]
    data[key] = msgs
    _save_history(data)


def clear_history(user_id: int):
    data = _load_history()
    data.pop(str(user_id), None)
    _save_history(data)


# ── Gemini client ──────────────────────────────────────────────────────────────
gemini_api_key = os.environ.get("GEMINI_API_KEY")
if gemini_api_key:
    ai_client = genai.Client(api_key=gemini_api_key)
    logger.info("Gemini AI ready")
else:
    ai_client = None
    logger.warning("GEMINI_API_KEY not set — AI responses disabled")


async def ask_gemini(user_id: int, display_name: str, question: str) -> str:
    history = get_history(user_id)
    contents = []
    for msg in history:
        contents.append(
            types.Content(role=msg["role"], parts=[types.Part(text=msg["text"])])
        )
    contents.append(
        types.Content(
            role="user", parts=[types.Part(text=f"{display_name} says: {question}")]
        )
    )

    response = await ai_client.aio.models.generate_content(
        model="gemini-2.0-flash",
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT, max_output_tokens=500
        ),
    )
    bot_reply = response.text.strip()
    append_history(user_id, f"{display_name} says: {question}", bot_reply)
    return bot_reply


async def ask_gemini_raw(prompt: str, max_tokens: int = 400) -> str:
    """One-shot Gemini call with no history (for game generation)."""
    response = await ai_client.aio.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt,
        config=types.GenerateContentConfig(max_output_tokens=max_tokens),
    )
    return response.text.strip()


# ── Minigame state (in-memory, resets on restart) ─────────────────────────────
# active_games[user_id] = { "type": "guess"|"hangman"|"trivia", ... }
active_games: dict[int, dict] = {}

HANGMAN_WORDS = [
    "elephant",
    "rainbow",
    "volcano",
    "galaxy",
    "tornado",
    "diamond",
    "lantern",
    "compass",
    "crystal",
    "dolphin",
    "penguin",
    "pyramid",
    "mushroom",
    "cactus",
    "blanket",
    "popcorn",
    "sunflower",
    "treasure",
    "blizzard",
    "coconut",
    "firefly",
    "glacier",
    "hammock",
    "jellyfish",
    "kayak",
    "leopard",
    "mango",
    "noodles",
    "ostrich",
    "porcupine",
    "quicksand",
    "raccoon",
    "saffron",
    "tangerine",
    "umbrella",
    "walrus",
    "xylophone",
    "yogurt",
    "zeppelin",
    "avocado",
    "broccoli",
    "caramel",
    "daffodil",
    "emerald",
    "flamingo",
    "goblin",
    "horizon",
    "igloo",
]

MAX_HANGMAN_WRONG = 6

HANGMAN_ART = [
    "```\n  +---+\n      |\n      |\n      |\n      |\n      |\n=========```",
    "```\n  +---+\n  O   |\n      |\n      |\n      |\n      |\n=========```",
    "```\n  +---+\n  O   |\n  |   |\n      |\n      |\n      |\n=========```",
    "```\n  +---+\n  O   |\n /|   |\n      |\n      |\n      |\n=========```",
    "```\n  +---+\n  O   |\n /|\\  |\n      |\n      |\n      |\n=========```",
    "```\n  +---+\n  O   |\n /|\\  |\n /    |\n      |\n      |\n=========```",
    "```\n  +---+\n  O   |\n /|\\  |\n / \\  |\n      |\n      |\n=========```",
]


def hangman_display(game: dict) -> str:
    word = game["word"]
    guessed: set = game["guessed"]
    wrong: int = game["wrong"]
    wrong_letters = sorted(l for l in guessed if l not in word)

    revealed = " ".join(c if c in guessed else "_" for c in word)
    art = HANGMAN_ART[wrong]
    wrong_str = (
        f"Wrong letters: **{', '.join(wrong_letters)}**"
        if wrong_letters
        else "No wrong guesses yet!"
    )

    return (
        f"{art}\n"
        f"**Word:** {revealed}\n"
        f"{wrong_str}\n"
        f"Lives left: {'❤️' * (MAX_HANGMAN_WRONG - wrong)}{'🖤' * wrong}"
    )


# ── Misc ───────────────────────────────────────────────────────────────────────
EIGHT_BALL_RESPONSES = [
    "It is certain.",
    "It is decidedly so.",
    "Without a doubt.",
    "Yes, definitely.",
    "You may rely on it.",
    "As I see it, yes.",
    "Most likely.",
    "Outlook good.",
    "Yes.",
    "Signs point to yes.",
    "Reply hazy, try again.",
    "Ask again later.",
    "Better not tell you now.",
    "Cannot predict now.",
    "Concentrate and ask again.",
    "Don't count on it.",
    "My reply is no.",
    "My sources say no.",
    "Outlook not so good.",
    "Very doubtful.",
]


class HealthHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"OK")

    def log_message(self, format, *args):
        pass


def start_health_server():
    port = int(os.environ.get("PORT", "8080"))
    server = http.server.HTTPServer(("0.0.0.0", port), HealthHandler)
    logger.info(f"Health server listening on port {port}")
    server.serve_forever()


# ── Bot ────────────────────────────────────────────────────────────────────────
intents = discord.Intents.default()
intents.message_content = True


class TuskBot(discord.Client):
    def __init__(self):
        super().__init__(intents=intents)
        self.tree = app_commands.CommandTree(self)

    async def setup_hook(self):
        await self.tree.sync()
        logger.info("Slash commands synced globally")

    async def on_ready(self):
        logger.info(f"Logged in as {self.user} (ID: {self.user.id})")
        await self.change_presence(
            activity=discord.Activity(
                type=discord.ActivityType.watching, name="for /tusk commands"
            )
        )

    async def on_message(self, message: discord.Message):
        if message.author.bot:
            return

        content = message.content.strip()
        content_lower = content.lower()
        uid = message.author.id

        # ── Active game input handling ──────────────────────────────────────
        if uid in active_games:
            game = active_games[uid]

            # Only handle game inputs in the channel the game was started in
            if message.channel.id == game["channel_id"]:
                if game["type"] == "guess":
                    if content.lstrip("-").isdigit():
                        guess = int(content)
                        target = game["number"]
                        game["attempts"] += 1
                        if guess < 1 or guess > 100:
                            await message.reply(
                                "Please guess a number between **1 and 100**! 🐘"
                            )
                        elif guess == target:
                            attempts = game["attempts"]
                            del active_games[uid]
                            medal = (
                                "🥇"
                                if attempts <= 5
                                else "🥈"
                                if attempts <= 10
                                else "🥉"
                            )
                            await message.reply(
                                f"🎉🎉 **YES!! {guess} is correct!!** {medal}\n"
                                f"You got it in **{attempts} attempt{'s' if attempts != 1 else ''}**!! Amazing!! 🐘✨"
                            )
                        elif guess < target:
                            await message.reply(
                                f"📈 Too low! Try **higher**! (Attempt #{game['attempts']}) 🔼"
                            )
                        else:
                            await message.reply(
                                f"📉 Too high! Try **lower**! (Attempt #{game['attempts']}) 🔽"
                            )
                        return

                elif game["type"] == "hangman":
                    letter = content_lower.strip()
                    if len(letter) == 1 and letter.isalpha():
                        word = game["word"]
                        guessed: set = game["guessed"]

                        if letter in guessed:
                            await message.reply(
                                f"You already guessed **{letter}**! Try a different letter 🐘"
                            )
                            return

                        guessed.add(letter)
                        if letter not in word:
                            game["wrong"] += 1

                        display = hangman_display(game)

                        # Check win
                        if all(c in guessed for c in word):
                            del active_games[uid]
                            await message.reply(
                                f"{display}\n\n"
                                f"🎉 **YOU WON!!** The word was **{word.upper()}**!! Amazing!! 🐘🌟"
                            )
                        # Check loss
                        elif game["wrong"] >= MAX_HANGMAN_WRONG:
                            del active_games[uid]
                            await message.reply(
                                f"{display}\n\n"
                                f"💀 **Game over!** The word was **{word.upper()}**! Don't give up — try again with `/tusk hangman`! 🐘"
                            )
                        else:
                            hint = (
                                "✅ Good guess!"
                                if letter in word
                                else f"❌ Nope, no **{letter}**!"
                            )
                            await message.reply(f"{hint}\n\n{display}")
                        return
                    elif len(letter) == len(game["word"]) and letter.isalpha():
                        # Full word guess
                        if letter == game["word"]:
                            word = game["word"]
                            del active_games[uid]
                            await message.reply(
                                f"🎉🎉 **INCREDIBLE!!** You guessed the whole word: **{word.upper()}**!! 🐘🌟✨"
                            )
                        else:
                            game["wrong"] += 1
                            display = hangman_display(game)
                            if game["wrong"] >= MAX_HANGMAN_WRONG:
                                word = game["word"]
                                del active_games[uid]
                                await message.reply(
                                    f"❌ Wrong! {display}\n\n💀 **Game over!** The word was **{word.upper()}**! Try `/tusk hangman` again! 🐘"
                                )
                            else:
                                await message.reply(
                                    f"❌ Wrong word guess!\n\n{display}"
                                )
                        return

                elif game["type"] == "trivia":
                    ans = content_lower.strip()
                    if ans in ("a", "b", "c", "d"):
                        correct = game["answer"]
                        correct_text = game["correct_text"]
                        del active_games[uid]
                        if ans == correct:
                            await message.reply(
                                f"🎉 **CORRECT!!** The answer was **{correct.upper()}**: {correct_text}!! "
                                f"You're so smart!! 🧠✨🐘"
                            )
                        else:
                            await message.reply(
                                f"❌ Not quite! The correct answer was **{correct.upper()}**: {correct_text}. "
                                f"Better luck next time!! 🐘💪"
                            )
                        return

        # ── Normal message handling ─────────────────────────────────────────
        mentioned = self.user in message.mentions
        name_used = "tushiny" in content_lower or "tusk" in content_lower

        if mentioned or name_used:
            question = content
            for mention in message.mentions:
                question = question.replace(f"<@{mention.id}>", "").replace(
                    f"<@!{mention.id}>", ""
                )
            for name in ("tushiny", "tusk", "Tushiny", "Tusk", "TUSHINY", "TUSK"):
                question = question.replace(name, "")
            question = question.strip(" ,!?.")

            if question and ai_client:
                async with message.channel.typing():
                    try:
                        reply = await ask_gemini(
                            uid, message.author.display_name, question
                        )
                        await message.reply(reply[:1900])
                    except Exception as e:
                        logger.error(f"Gemini error: {e}")
                        await message.reply(
                            "Sorry, I had trouble thinking of a response 😅"
                        )
            elif not question:
                await message.reply(
                    f"Heyyy {message.author.mention}!! 🎉 So happy to see you! "
                    "Ask me anything or try `/tusk help` 🐘✨"
                )
            else:
                await message.reply(
                    "Omg did someone say Tusk?! That's ME!! 🐘🎉 Ask me something, I'd love to help!"
                )
            return

        if content_lower in ("hello", "hi", "hey", "sup", "yo"):
            await message.reply(
                f"HEYYY {message.author.mention}!! 🎉 So great to see you! Hope you're having an amazing day! 🌟"
            )
        elif content_lower in ("good bot", "good job", "nice bot"):
            await message.reply(
                "THANK YOU SO MUCH!! 🥹💖 That genuinely made my day!! You're amazing!! 🐘✨"
            )
        elif content_lower == "bad bot":
            await message.reply(
                "Aww I'll do better next time, I promise!! 🥺💪 You got this and so do I! 🌟"
            )


bot = TuskBot()
tusk = app_commands.Group(name="tusk", description="Tusk bot commands")


# ── Utility commands ───────────────────────────────────────────────────────────
@tusk.command(name="ping", description="Check the bot latency")
async def ping(interaction: discord.Interaction):
    await interaction.response.send_message(
        f"Pong! 🏓 Latency: **{round(bot.latency * 1000)}ms**"
    )


@tusk.command(name="serverinfo", description="Show info about this server")
async def serverinfo(interaction: discord.Interaction):
    guild = interaction.guild
    if not guild:
        await interaction.response.send_message(
            "This command can only be used in a server.", ephemeral=True
        )
        return
    embed = discord.Embed(title=guild.name, color=discord.Color.blurple())
    if guild.icon:
        embed.set_thumbnail(url=guild.icon.url)
    embed.add_field(name="👑 Owner", value=f"<@{guild.owner_id}>", inline=True)
    embed.add_field(name="👥 Members", value=str(guild.member_count), inline=True)
    embed.add_field(name="💬 Channels", value=str(len(guild.channels)), inline=True)
    embed.add_field(name="🎭 Roles", value=str(len(guild.roles)), inline=True)
    embed.add_field(name="🌍 Region", value=str(guild.preferred_locale), inline=True)
    embed.add_field(
        name="📅 Created",
        value=f"<t:{int(guild.created_at.timestamp())}:D>",
        inline=True,
    )
    embed.set_footer(text=f"Server ID: {guild.id}")
    await interaction.response.send_message(embed=embed)


@tusk.command(name="userinfo", description="Show info about a user")
@app_commands.describe(user="The user to look up (leave blank for yourself)")
async def userinfo(interaction: discord.Interaction, user: discord.Member = None):
    target = user or interaction.user
    embed = discord.Embed(
        title=str(target),
        color=target.color
        if isinstance(target, discord.Member)
        else discord.Color.blurple(),
    )
    embed.set_thumbnail(url=target.display_avatar.url)
    embed.add_field(name="🆔 ID", value=str(target.id), inline=True)
    embed.add_field(name="🤖 Bot", value="Yes" if target.bot else "No", inline=True)
    embed.add_field(
        name="📅 Joined Discord",
        value=f"<t:{int(target.created_at.timestamp())}:D>",
        inline=True,
    )
    if isinstance(target, discord.Member) and target.joined_at:
        embed.add_field(
            name="📥 Joined Server",
            value=f"<t:{int(target.joined_at.timestamp())}:D>",
            inline=True,
        )
        top_role = target.top_role
        if top_role != interaction.guild.default_role:
            embed.add_field(name="🎭 Top Role", value=top_role.mention, inline=True)
    await interaction.response.send_message(embed=embed)


@tusk.command(name="8ball", description="Ask the magic 8-ball a question")
@app_commands.describe(question="Your yes/no question")
async def eight_ball(interaction: discord.Interaction, question: str):
    answer = random.choice(EIGHT_BALL_RESPONSES)
    embed = discord.Embed(color=discord.Color.dark_purple())
    embed.add_field(name="❓ Question", value=question, inline=False)
    embed.add_field(name="🎱 Answer", value=answer, inline=False)
    await interaction.response.send_message(embed=embed)


@tusk.command(
    name="forget", description="Make Tushiny forget your entire conversation history"
)
async def forget(interaction: discord.Interaction):
    clear_history(interaction.user.id)
    await interaction.response.send_message(
        "Done! 🧹✨ I've wiped my memory of our conversations — fresh start!! 🐘💖",
        ephemeral=True,
    )


@tusk.command(
    name="history", description="See how many messages Tushiny remembers from you"
)
async def history_cmd(interaction: discord.Interaction):
    msgs = get_history(interaction.user.id)
    exchanges = len(msgs) // 2
    embed = discord.Embed(
        title="🧠 Your Conversation Memory", color=discord.Color.green()
    )
    if exchanges == 0:
        embed.description = "I don't remember any conversations with you yet! Talk to me by mentioning my name 🐘"
    else:
        embed.description = (
            f"I remember **{exchanges} exchange{'s' if exchanges != 1 else ''}** with you "
            f"({len(msgs)}/{MAX_HISTORY_PER_USER} messages stored).\n\n"
            "I use all of this when I reply to you, so I know exactly what we've talked about! 🌟"
        )
    embed.set_footer(text="Use /tusk forget to clear your history")
    await interaction.response.send_message(embed=embed, ephemeral=True)


# ── 🎮 Minigames ───────────────────────────────────────────────────────────────


@tusk.command(name="rps", description="🎮 Play Rock Paper Scissors against Tushiny!")
@app_commands.describe(choice="Pick your move!")
@app_commands.choices(
    choice=[
        app_commands.Choice(name="🪨 Rock", value="rock"),
        app_commands.Choice(name="📄 Paper", value="paper"),
        app_commands.Choice(name="✂️ Scissors", value="scissors"),
    ]
)
async def rps(interaction: discord.Interaction, choice: app_commands.Choice[str]):
    moves = ["rock", "paper", "scissors"]
    emojis = {"rock": "🪨", "paper": "📄", "scissors": "✂️"}
    bot_move = random.choice(moves)
    player = choice.value

    if player == bot_move:
        result = "🤝 It's a **tie**!! Great minds think alike!! 🐘"
    elif (
        (player == "rock" and bot_move == "scissors")
        or (player == "paper" and bot_move == "rock")
        or (player == "scissors" and bot_move == "paper")
    ):
        result = "🎉 **You WIN!!** Woohoo!! You're incredible!! 🌟🐘"
    else:
        result = "😄 **Tushiny wins!!** Hehe!! Better luck next time!! 🐘✨"

    embed = discord.Embed(title="🎮 Rock Paper Scissors!", color=discord.Color.gold())
    embed.add_field(
        name=f"You chose",
        value=f"{emojis[player]} **{player.capitalize()}**",
        inline=True,
    )
    embed.add_field(
        name=f"Tushiny chose",
        value=f"{emojis[bot_move]} **{bot_move.capitalize()}**",
        inline=True,
    )
    embed.add_field(name="Result", value=result, inline=False)
    await interaction.response.send_message(embed=embed)


@tusk.command(name="guess", description="🎮 Start a number guessing game (1–100)!")
async def guess(interaction: discord.Interaction):
    uid = interaction.user.id
    if uid in active_games:
        game = active_games[uid]
        await interaction.response.send_message(
            f"You already have an active **{game['type']}** game! Finish it first, or type `quit` to cancel 🐘",
            ephemeral=True,
        )
        return

    number = random.randint(1, 100)
    active_games[uid] = {
        "type": "guess",
        "number": number,
        "attempts": 0,
        "channel_id": interaction.channel_id,
    }

    embed = discord.Embed(
        title="🔢 Number Guessing Game!",
        description=(
            "I'm thinking of a number between **1 and 100**! 🤔\n\n"
            "Just type your guess in this channel!\n"
            "I'll tell you if you're too high or too low 🐘✨\n\n"
            "_(Type `quit` to give up)_"
        ),
        color=discord.Color.blue(),
    )
    await interaction.response.send_message(embed=embed)


@tusk.command(name="hangman", description="🎮 Play Hangman — guess the hidden word!")
async def hangman(interaction: discord.Interaction):
    uid = interaction.user.id
    if uid in active_games:
        game = active_games[uid]
        await interaction.response.send_message(
            f"You already have an active **{game['type']}** game! Finish it first, or type `quit` to cancel 🐘",
            ephemeral=True,
        )
        return

    word = random.choice(HANGMAN_WORDS)
    active_games[uid] = {
        "type": "hangman",
        "word": word,
        "guessed": set(),
        "wrong": 0,
        "channel_id": interaction.channel_id,
    }

    display = hangman_display(active_games[uid])
    embed = discord.Embed(
        title="🪓 Hangman!",
        description=(
            f"{display}\n\n"
            f"The word has **{len(word)} letters**!\n"
            "Type a **letter** to guess, or type the **whole word** if you know it!\n"
            "_(Type `quit` to give up)_"
        ),
        color=discord.Color.orange(),
    )
    await interaction.response.send_message(embed=embed)


@tusk.command(name="trivia", description="🎮 Answer an AI-generated trivia question!")
async def trivia(interaction: discord.Interaction):
    uid = interaction.user.id
    if uid in active_games:
        game = active_games[uid]
        await interaction.response.send_message(
            f"You already have an active **{game['type']}** game! Finish it first, or type `quit` to cancel 🐘",
            ephemeral=True,
        )
        return

    if not ai_client:
        await interaction.response.send_message(
            "AI isn't available right now 😅", ephemeral=True
        )
        return

    await interaction.response.defer()

    try:
        raw = await ask_gemini_raw(
            "Generate a fun trivia question with 4 multiple choice options (A, B, C, D). "
            "Format EXACTLY like this with no extra text:\n"
            "QUESTION: <question text>\n"
            "A: <option>\n"
            "B: <option>\n"
            "C: <option>\n"
            "D: <option>\n"
            "ANSWER: <A or B or C or D>\n"
            "Make it fun and interesting — mix topics like history, science, pop culture, nature, food!"
        )

        lines = {
            line.split(":")[0].strip(): ":".join(line.split(":")[1:]).strip()
            for line in raw.splitlines()
            if ":" in line
        }

        question_text = lines.get("QUESTION", "")
        a = lines.get("A", "")
        b = lines.get("B", "")
        c = lines.get("C", "")
        d = lines.get("D", "")
        answer = lines.get("ANSWER", "").upper().strip().replace(".", "")

        if not question_text or answer not in ("A", "B", "C", "D"):
            await interaction.followup.send(
                "Hmm, I couldn't make a good question! Try again 🐘"
            )
            return

        correct_texts = {"A": a, "B": b, "C": c, "D": d}
        active_games[uid] = {
            "type": "trivia",
            "answer": answer.lower(),
            "correct_text": correct_texts[answer],
            "channel_id": interaction.channel_id,
        }

        embed = discord.Embed(
            title="🧠 Trivia Time!!",
            description=f"**{question_text}**",
            color=discord.Color.purple(),
        )
        embed.add_field(name="A", value=a, inline=False)
        embed.add_field(name="B", value=b, inline=False)
        embed.add_field(name="C", value=c, inline=False)
        embed.add_field(name="D", value=d, inline=False)
        embed.set_footer(
            text="Type A, B, C, or D in chat to answer! • Type quit to skip"
        )
        await interaction.followup.send(embed=embed)

    except Exception as e:
        logger.error(f"Trivia error: {e}")
        await interaction.followup.send(
            "Oops, something went wrong generating the question 😅 Try again!"
        )


@tusk.command(
    name="wouldyourather", description="🎮 Would you rather...? (AI-generated!)"
)
async def would_you_rather(interaction: discord.Interaction):
    if not ai_client:
        await interaction.response.send_message(
            "AI isn't available right now 😅", ephemeral=True
        )
        return

    await interaction.response.defer()
    try:
        raw = await ask_gemini_raw(
            "Generate a fun, creative, and safe-for-all-ages 'Would You Rather' question. "
            "Format EXACTLY like this with no extra text:\n"
            "OPTION_A: <first option>\n"
            "OPTION_B: <second option>\n"
            "Make both options funny, interesting, and hard to choose between!"
        )
        lines = {
            line.split(":")[0].strip(): ":".join(line.split(":")[1:]).strip()
            for line in raw.splitlines()
            if ":" in line
        }
        option_a = lines.get("OPTION_A", "")
        option_b = lines.get("OPTION_B", "")

        if not option_a or not option_b:
            await interaction.followup.send(
                "Hmm, I couldn't think of one! Try again 🐘"
            )
            return

        embed = discord.Embed(
            title="🤔 Would You Rather...?",
            color=discord.Color.pink()
            if hasattr(discord.Color, "pink")
            else discord.Color.magenta(),
        )
        embed.add_field(name="🅰️ Option A", value=option_a, inline=False)
        embed.add_field(name="🅱️ Option B", value=option_b, inline=False)
        embed.set_footer(
            text="React or reply with A or B — debate with your friends! 🐘"
        )
        await interaction.followup.send(embed=embed)

    except Exception as e:
        logger.error(f"WYR error: {e}")
        await interaction.followup.send("Oops, something went wrong! Try again 🐘")


# ── Help ───────────────────────────────────────────────────────────────────────
@tusk.command(name="help", description="List all Tushiny bot commands")
async def help_cmd(interaction: discord.Interaction):
    embed = discord.Embed(
        title="🐘 Tushiny Bot Commands",
        description="All commands start with `/tusk`",
        color=discord.Color.blurple(),
    )
    embed.add_field(
        name="🔧 Utility",
        value=(
            "`/tusk ping` — Latency check\n"
            "`/tusk serverinfo` — Server stats\n"
            "`/tusk userinfo [user]` — User stats\n"
            "`/tusk 8ball <question>` — Magic 8-ball\n"
            "`/tusk history` — Conversation memory count\n"
            "`/tusk forget` — Clear your memory"
        ),
        inline=False,
    )
    embed.add_field(
        name="🎮 Minigames",
        value=(
            "`/tusk rps` — Rock Paper Scissors 🪨📄✂️\n"
            "`/tusk guess` — Number guessing (1–100) 🔢\n"
            "`/tusk hangman` — Hangman 🪓\n"
            "`/tusk trivia` — AI trivia question 🧠\n"
            "`/tusk wouldyourather` — Would You Rather 🤔"
        ),
        inline=False,
    )
    embed.set_footer(text="Tushiny Bot 🐘 — mention me or say my name to chat!")
    await interaction.response.send_message(embed=embed)


bot.tree.add_command(tusk)


# ── Quit handler (extra on_message check via monkey-patch trick) ──────────────
_original_on_message = TuskBot.on_message


async def _patched_on_message(self, message: discord.Message):
    if not message.author.bot and message.content.strip().lower() == "quit":
        uid = message.author.id
        if uid in active_games:
            game = active_games.pop(uid)
            await message.reply(
                f"Game cancelled! 👋 Your **{game['type']}** game has ended 🐘"
            )
            return
    await _original_on_message(self, message)


TuskBot.on_message = _patched_on_message


def main():
    token = os.environ.get("DISCORD_TOKEN")
    if not token:
        raise ValueError("DISCORD_TOKEN environment variable is required")

    threading.Thread(target=start_health_server, daemon=True).start()
    bot.run(token, log_handler=None)


if __name__ == "__main__":
    main()
