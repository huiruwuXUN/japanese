from PIL import Image, ImageFont, ImageDraw
import os

def textsize(text, font):
    im = Image.new(mode="P", size=(0, 0))
    draw = ImageDraw.Draw(im)
    _, _, width, height = draw.textbbox((0, 0), text=text, font=font)
    return width, height

def font_extract(font_name):
    font_path = os.path.join('./character_classifying_cnn/fonts', font_name)
    kanji_dir = './character_classifying_cnn/outputs/images/kanji/'
    katakana_dir = './character_classifying_cnn/outputs/images/katakana/'
    hiragana_dir = './character_classifying_cnn/outputs/images/hiragana/'
    font_size = 200
    image_size = 256
    characters = '愛安暗医委意育員駅液運映英栄永衛駅園演縁遠泳往応桜可価画会海火花貨過快解顔感期希季紀基寄規喜器帰起記客急級休吸供競共協鏡競極計警劇権源厳己語呼後子好効広考行高黄光公工康交構考行合国黒今才材在再最坂作昨策山賛正生政成性姓'  
    hiragana = 'あいうえおかきくけこさしすせそたちつてとなにぬねのはひふへほまみむめもやゆよらりるれろわをん'
    katakana = 'アイウエオカキクケコサシスセソタチツテトナニヌネノハヒフヘホマミムメモヤユヨラリルレロワヲン'
    # kana = hiragana + katakana
    if not os.path.exists(kanji_dir):
        os.makedirs(kanji_dir)
    if not os.path.exists(katakana_dir):
        os.makedirs(katakana_dir)
    if not os.path.exists(hiragana_dir):
        os.makedirs(hiragana_dir)

    print(os.path.exists(font_path))

    font = ImageFont.truetype(font_path, font_size)

    for char in characters:
        image = Image.new('RGB', (image_size, image_size), color = (255, 255, 255))
        draw = ImageDraw.Draw(image)
        text_width, text_height = textsize(char, font=font)
        text_x = (font_size - text_width) / 2
        text_y = (font_size - text_height) / 2
        draw.text((text_x, text_y), char, font=font, fill=(0, 0, 0))
        
        output_path = os.path.join(kanji_dir, f'{char}_{font_name}.png')
        image.save(output_path)

    for char in hiragana:
        image = Image.new('RGB', (image_size, image_size), color = (255, 255, 255))
        draw = ImageDraw.Draw(image)
        text_width, text_height = textsize(char, font=font)
        text_x = (font_size - text_width) / 2
        text_y = (font_size - text_height) / 2
        draw.text((text_x, text_y), char, font=font, fill=(0, 0, 0))
        
        output_path = os.path.join(hiragana_dir, f'{char}_{font_name}.png')
        image.save(output_path)

    for char in katakana:
        image = Image.new('RGB', (image_size, image_size), color = (255, 255, 255))
        draw = ImageDraw.Draw(image)
        text_width, text_height = textsize(char, font=font)
        text_x = (font_size - text_width) / 2
        text_y = (font_size - text_height) / 2
        draw.text((text_x, text_y), char, font=font, fill=(0, 0, 0))
        
        output_path = os.path.join(katakana_dir, f'{char}_{font_name}.png')
        image.save(output_path)

    print("Finish font extraction")

if __name__ == "__main__":
    fonts = ['1.ttf', '3.ttf', '4.ttf', '6.ttf', 'YujiMai-Regular.ttf']
    for font in fonts:
        font_extract(font)
