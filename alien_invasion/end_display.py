import pygame
from pygame.event import EventType

validChars = "`1234567890-=qwertyuiop[]\\asdfghjkl;'zxcvbnm,./"
shiftChars = '~!@#$%^&*()_+QWERTYUIOP{}|ASDFGHJKL:"ZXCVBNM<>?'
shiftDown = False

class TextBox(pygame.sprite.Sprite):
  def __init__(self):
    pygame.sprite.Sprite.__init__(self)
    self.text = ""
    self.font = pygame.font.Font('fonts/RussoOne.ttf', 40)
    self.image = self.font.render("Enter your name to update highscore ranking", False, [255, 255, 255])
    self.rect = self.image.get_rect()
    self.bg_color = (10, 5, 50)

  def add_chr(self, char):
    global shiftDown
    if char in validChars and not shiftDown:
        self.text += char
    elif char in validChars and shiftDown:
        self.text += shiftChars[validChars.index(char)]
    self.update()

  def update(self):
    old_rect_pos = self.rect.center
    self.image = self.font.render(self.text, False, [255, 255, 255])
    self.rect = self.image.get_rect()
    self.rect.center = old_rect_pos

def ask(screen):
    textBox = TextBox()
    shiftDown = False
    textBox.rect.center = [520, 240]

    running = True
    while running:
        screen.fill([10, 5, 50])
        screen.blit(textBox.image, textBox.rect)
        pygame.display.flip()
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False
            if e.type == pygame.KEYUP:
                if e.key in [pygame.K_RSHIFT, pygame.K_LSHIFT]:
                    shiftDown = False
            if e.type == pygame.KEYDOWN:
                textBox.add_chr(pygame.key.name(e.key))
                if e.key == pygame.K_SPACE:
                    textBox.text += " "
                    textBox.update()
                if e.key in [pygame.K_RSHIFT, pygame.K_LSHIFT]:
                    shiftDown = True
                if e.key == pygame.K_BACKSPACE:
                    textBox.text = textBox.text[:-1]
                    textBox.update()
                if e.key == pygame.K_RETURN:
                    if len(textBox.text) > 0:                    
                        return textBox.text
                    else:
                        return ""

