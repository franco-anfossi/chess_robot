import pygame
from piezas import Piezas
from parametros import FILAS, COLUMNAS, ARISTA_IMG

class Juego:
    def __init__(self):
        self.piezas = Piezas()

    def mostrar_fondo(self, screen):
        for fila in range(FILAS):
            for columna in range(COLUMNAS):
                if (fila + columna) % 2 != 0:
                    color = (165, 117, 80) # oscuro
                else:
                    color = (235, 209, 166) # claro

                cuadros = (columna * ARISTA_IMG, fila * ARISTA_IMG, ARISTA_IMG, ARISTA_IMG)
                pygame.draw.rect(screen, color, cuadros)

    def mostrar_piezas(self, screen, board):
        fila = 0
        columna = 0
        for pieza in board.fen().split()[0]:  # estado del tablero
            if pieza == "/":
                fila += 1
                columna = 0

            elif pieza in ('1', '2', '3', '4', '5', '6', '7', '8'):
                columna += int(pieza) # avanza el número de casillas
            
            elif pieza in ('K', 'k', 'Q', 'q', 'R', 'r', 'N', 'n', 'B', 'b', 'P', 'p'):
                self.piezas.agregar_piezas(screen, pieza, columna * ARISTA_IMG, fila * ARISTA_IMG)
                columna += 1

        pygame.display.update()