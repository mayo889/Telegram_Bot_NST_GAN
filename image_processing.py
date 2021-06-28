from model_cyclegan import CycleGAN
from model_nst import StyleTransfer
from aiogram import Bot
from os import environ
import keyboards as kb
import logging
import warnings
warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO)

BOT_TOKEN = environ.get("BOT_TOKEN")


async def cycle_gan(message, image, type_algo):
    if type_algo == 'summer2winter':
        wts_path = "models_wts/summer2winter.pth"
    elif type_algo == 'winter2summer':
        wts_path = "models_wts/winter2summer.pth"

    new_image = CycleGAN.run_gan(wts_path, image)

    logging.info(f"Finished CycleGAN")

    tmp_bot = Bot(token=BOT_TOKEN)
    await tmp_bot.send_photo(message.chat.id, photo=new_image)
    await tmp_bot.send_message(message.chat.id, "Надеюсь, тебе понравилось.\n\n Хочешь попробовать еще раз?",
                               reply_markup=kb.algo_keyboard())
    await tmp_bot.close()


async def style_transfer(message, style_image, content_image):
    new_image = StyleTransfer.run_nst(style_image, content_image)

    logging.info(f"Finished Style Transfer")

    tmp_bot = Bot(token=BOT_TOKEN)
    await tmp_bot.send_photo(message.chat.id, photo=new_image)
    await tmp_bot.send_message(message.chat.id, "Надеюсь, тебе понравилось.\n\n Хочешь попробовать еще раз?",
                               reply_markup=kb.algo_keyboard())
    await tmp_bot.close()
