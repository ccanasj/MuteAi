import discord
from discord.ext import commands
import cv2
from discord.ext.commands.core import after_invoke
import numpy as np
from io import BytesIO
from PIL import Image
from ModeloCNN import predict_image
import asyncio
import Mongo


class Bot(commands.Bot):
    def __init__(self):
        super().__init__(command_prefix=commands.when_mentioned_or("$"),case_insensitive=True)

    async def on_ready(self):
        print(f"Logged in as {self.user} (ID: {self.user.id})")
        print("----------------------------------------------")

bot = Bot()
bot.remove_command('help')
Classes = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
Letras = list('🇦🇧🇨🇩🇪🇫🇬🇭🇮🇯🇰🇱🇲🇳🇴🇵🇶🇷🇸🇹🇺🇻🇼🇽🇾🇿')

@bot.command()
async def help(ctx):
    embed = discord.Embed(color = discord.Colour.gold())
    embed.set_author(name = 'Sign Translator commands')
    embed.add_field(name="Funciones", 
    value='**$traducir** (Traduce la imagen con una seña que adjuntes) \n**$galeria** (Ver tus imagenes guardadas) \n**$sign** (Ver señas posibles)\n**$flip** (Girar las imagen que adjuntes)')
    await ctx.send(embed = embed)

async def Decision(usuario,message):
    await message.add_reaction('✅')
    await message.add_reaction('↩')
    def check(reaction, user):
        return user == usuario and str(reaction.emoji) in ('↩','✅')
    try:
        reaction, user = await bot.wait_for('reaction_add', timeout=15.0, check=check)
    except asyncio.TimeoutError:
        await message.clear_reactions()
        return None
    else:
        return True if reaction.emoji == '✅' else False

@bot.command()
@after_invoke(Mongo.CerrarDB)
@commands.max_concurrency(1,per = commands.BucketType.user)
async def traducir(ctx):
    files = ctx.message.attachments
    if files:
        archivo = await files[0].read()
        pil_image = Image.open(BytesIO(archivo))
        opencvImage = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        res = cv2.resize(opencvImage, dsize=(28, 28), interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
        numpydata = np.asarray(gray, dtype="float32")
        predicciones = predict_image(numpydata)
        letras = [Classes[prediccion.item()] for prediccion in predicciones]
        Embed = discord.Embed(title = 'Traductor alfabeto lenguaje de señas')
        Embed.add_field(name = 'Traduccion mas probable', value = f'**{Letras[predicciones[0].item()]}**')
        Embed.add_field(name = 'Traducciones posibles', value = f'**{letras[1]} - {letras[2]}**',inline = False)
        Embed.set_image(url = files[0].url)
        Embed.set_footer(text = 'Si no genera lo que deseabas puedes intentar cambiar el fondo a blanco')
        mensaje = await ctx.send(embed = Embed)
        correcta = await Decision(ctx.author,mensaje)
        if correcta == None:
            return
        if correcta:
            await Mongo.GuardarUsuario(ctx.author.id,archivo,letras)
            await mensaje.clear_reactions()
            await ctx.send('Se ha guardado correctamente la imagen en tu galeria')
        else:
            flipHorizontal = cv2.flip(gray, 1)
            numpydata = np.asarray(flipHorizontal, dtype="float32")
            predicciones = predict_image(numpydata)
            img = pil_image.transpose(Image.FLIP_LEFT_RIGHT)
            letras = [Classes[prediccion.item()] for prediccion in predicciones]
            Embed.set_field_at(0,name = 'Traduccion mas probable', value = f'**{Letras[predicciones[0].item()]}**')
            Embed.set_field_at(1,name = 'Traducciones posibles', value = f'**{letras[1]} - {letras[2]}**',inline = False)
            with BytesIO() as image_binary:
                img.save(image_binary, 'PNG')
                image_binary.seek(0)
                chart = discord.File(fp=image_binary, filename = 'giro.png')
            Embed.set_image(url = "attachment://giro.png")
            mensaje = await ctx.send(embed = Embed ,file = chart)
            correcta = await Decision(ctx.author,mensaje)
            if correcta == None:
                return
            if correcta:
                await Mongo.GuardarUsuario(ctx.author.id,archivo,letras)
                await ctx.send('Se ha guardado correctamente la imagen en tu galeria')
            await mensaje.clear_reactions()
    else:
        await ctx.send('No adjuntaste ningun archivo')

@bot.command()
@commands.max_concurrency(1,per = commands.BucketType.user)
async def flip(ctx):
    files = ctx.message.attachments
    if files:
        archivo = await files[0].read()
        pil_image = Image.open(BytesIO(archivo))
        img = pil_image.transpose(Image.FLIP_LEFT_RIGHT)
        with BytesIO() as image_binary:
            img.save(image_binary, 'PNG')
            image_binary.seek(0)
            await ctx.send(file=discord.File(fp=image_binary, filename='giro.png'))
    else:
        await ctx.send('No adjuntaste ninguna imagen')

@bot.command()
@commands.max_concurrency(1,per = commands.BucketType.user)
async def sign(ctx):
    await ctx.send(file=discord.File('amer_sign2.png'))


@bot.command()
@after_invoke(Mongo.CerrarDB)
@commands.max_concurrency(1,per = commands.BucketType.user)
async def galeria(ctx):
    resultado = await Mongo.GetImagenes(ctx.author.id)
    if not resultado:
        await ctx.send('No tienes imagenes en tu galeria')
        return
    pagina = 0        
    embedVar = discord.Embed(title = "Galeria")
    embedVar.add_field(name = "Traduccion",value = resultado[0][1])
    with BytesIO() as image_binary:
        resultado[0][0].save(image_binary, 'PNG')
        image_binary.seek(0)
        chart = discord.File(fp=image_binary, filename = 'foto.png')
        image_binary.close()
    embedVar.set_image(url = "attachment://foto.png")
    embedVar.set_footer(text = f'Pagina {pagina + 1}/{len(resultado)}')
    mensaje = await ctx.send(embed = embedVar ,file = chart)
    await mensaje.add_reaction('⏩')
    terminar = False
    def check(reaction, user):
        return user == ctx.author and str(reaction.emoji) in ('⏩')
    while not terminar and len(resultado) > 1:
        try:
            reaction, user = await bot.wait_for('reaction_add', timeout=15.0, check=check)
        except asyncio.TimeoutError:
            terminar = True
            await mensaje.clear_reactions()
        else:
            await mensaje.delete()
            pagina += 1
            pagina = 0 if pagina >= len(resultado) else pagina
            embedVar.set_field_at(0,name = "Traduccion",value = resultado[pagina][1])
            with BytesIO() as image_binary:
                resultado[pagina][0].save(image_binary, 'PNG')
                image_binary.seek(0)
                chart = discord.File(fp=image_binary, filename='foto.png')
                image_binary.close()
            embedVar.set_image(url = "attachment://foto.png")
            embedVar.set_footer(text = f'Pagina {pagina + 1}/{len(resultado)}')
            mensaje = await ctx.send(embed = embedVar ,file = chart)
            await mensaje.add_reaction('⏩')

@bot.event
async def on_command_error(ctx,error):
    if isinstance(error,commands.CommandNotFound):
        await ctx.reply(f"Este comando no existe o lo invocaste mal")
    elif isinstance(error,commands.MaxConcurrencyReached):
        await ctx.reply('Acaba el comando anterior para seguir con este')
    elif isinstance(error,commands.BadArgument):
        await ctx.reply('Este no es un parametro correcto')
    else:
        raise error

bot.run("TOKEN")