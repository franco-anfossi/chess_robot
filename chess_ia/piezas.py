import pygame

class Piezas:
    def __init__(self):
        self.piezas = {
            "K": pygame.image.load("images/rey_blanco.png").convert_alpha(),
            "Q": pygame.image.load("images/reina_blanco.png").convert_alpha(),
            "R": pygame.image.load("images/torre_blanco.png").convert_alpha(),
            "B": pygame.image.load("images/alfil_blanco.png").convert_alpha(),
            "N": pygame.image.load("images/caballo_blanco.png").convert_alpha(),
            "P": pygame.image.load("images/peon_blanco.png").convert_alpha(),
            "k": pygame.image.load("images/rey_negro.png").convert_alpha(),
            "q": pygame.image.load("images/reina_negro.png").convert_alpha(),
            "r": pygame.image.load("images/torre_negro.png").convert_alpha(),
            "b": pygame.image.load("images/alfil_negro.png").convert_alpha(),
            "n": pygame.image.load("images/caballo_negro.png").convert_alpha(),
            "p": pygame.image.load("images/peon_negro.png").convert_alpha()
        }

        self.rects = {}
        self.escalar_imagenes()

    def agregar_piezas(self, screen, pieza, fila, columna):
        if (fila, columna) not in self.rects:  # Solo se crea un rect por cuadro
            rect = pygame.Rect(fila, columna, 100, 100)  # Crea un rectángulo 
            self.rects[(fila, columna)] = (pieza, rect)

        pieza, rect = self.rects[(fila, columna)]
        screen.blit(self.piezas[pieza], rect)

    def escalar_imagenes(self):
        for keys in self.piezas:
            self.piezas[keys] = pygame.transform.scale(self.piezas[keys], (100, 100))