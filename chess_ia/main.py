import pygame
import sys
import chess
from juego import Juego
from parametros import ALTO, ANCHO, ARISTA_IMG
from movimientos import Movimiento

class Main:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((ANCHO, ALTO))
        pygame.display.set_caption("Chess")
        self.juego = Juego()
        self.board = chess.Board()
        self.drag = None
        self.coords = {0: "a", 1: "b", 2: "c", 3: "d", 4: "e", 5: "f", 6: "g", 7: "h"}
        self.movimiento = Movimiento()

    def loop(self):
        continuar_partida = True
        continuar = True
        while continuar:

            self.juego.mostrar_fondo(self.screen)
            self.juego.mostrar_piezas(self.screen, self.board)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

                if continuar_partida:
                    if event.type == pygame.MOUSEBUTTONDOWN:
                        self.agarrar_pieza(event)

                    elif event.type == pygame.MOUSEMOTION:
                        if self.drag:
                            self.mover_pieza(event)

                    elif event.type == pygame.MOUSEBUTTONUP:
                        if self.drag:
                            self.soltar_pieza(event)

                        if self.movimiento.verificar_jaque_mate(self.board):
                            pygame.display.set_caption("Chess - Jaque mate")
                            continuar_partida = False

                        elif self.movimiento.verificar_tablas(self.board):
                            pygame.display.set_caption("Chess - Tablas")
                            continuar_partida = False

                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r:
                        self.board.reset()
                        self.juego = Juego()
                        continuar_partida = True
                        pygame.display.set_caption("Chess")

                pygame.display.update()

    # funciones de evento de mouse
    def agarrar_pieza(self, event):
        if event.button == 1:  # Botón izquierdo del ratón
            for (fila, columna), (pieza, rect) in self.juego.piezas.rects.items():
                if rect.collidepoint(event.pos):  # Se hace clic

                    is_negro= (pieza.isupper() and not self.board.turn)
                    is_blanco = (pieza.islower() and self.board.turn)
                    if is_blanco or is_negro:
                        return

                    event.pos = rect.center # clic al centro de la pieza
                    dif = (event.pos[0] - rect.centerx, event.pos[1] - rect.centery)
                    # Guarda en drag
                    self.drag = (pieza, rect, (fila, columna), dif)

                    break

    def mover_pieza(self, event):
        if self.drag:  # Si se está arrastrando una pieza
            pieza, rect, pos_inicial, dif = self.drag
            # Mueve la pieza sigue al mouse
            rect.center = (event.pos[0] - dif[0], event.pos[1] - dif[1])

    def soltar_pieza(self, event):
        if event.button == 1:
            if self.drag:  # Si se estaba arrastrando una pieza
                pieza, rect, pos_inicial, dif = self.drag

                # Calcula la fila y la columna de la pieza soltada
                fila = event.pos[0] // ARISTA_IMG
                columna = event.pos[1] // ARISTA_IMG

                # Validacion de movimiento
                inicio = f"{self.coords[pos_inicial[0] // 100] + str(8 - (pos_inicial[1] // 100))}"
                final = f"{self.coords[fila] + str(8 - columna)}"
                pieza_selec = None

                if ((pieza == "P" and final[1] == "8") or (pieza == "p" and final[1] == "1")):
                    pieza_selec, tupla = self.promocion_pieza(fila, columna, pieza)
                    final += pieza_selec

                validez = self.movimiento.verificar_movimiento(self.board, inicio, final)

                if validez:
                    # Calcula posicion
                    top = columna * ARISTA_IMG
                    left = fila * ARISTA_IMG

                    if pieza_selec != None:
                        if pieza.isupper():
                            self.juego.piezas.rects[(left, top)] = tupla
                        else:
                            self.juego.piezas.rects[(left, top)] = tupla
                    else:
                        # Ancla rect a la posición
                        rect.topleft = (left, top)
                        self.juego.piezas.rects[(left, top)] = (pieza, rect)
                        self.juego.piezas.rects.pop(pos_inicial)
                else:
                    # Ancla rect a la posición inicial
                    top = pos_inicial[1]
                    left = pos_inicial[0]
                    rect.topleft = (left, top)

                self.drag = None  
        pygame.display.update()

# --------------------------- promocion de peon ---------------------------
    def promocion_pieza(self, fila, columna, pieza):
        self.piezas_promocion_b = {
            "Q": pygame.image.load("images/reina_blanco.png").convert_alpha(),
            "R": pygame.image.load("images/torre_blanco.png").convert_alpha(),
            "B": pygame.image.load("images/alfil_blanco.png").convert_alpha(),
            "N": pygame.image.load("images/caballo_blanco.png").convert_alpha()
        }

        self.piezas_promocion_n = {
            "q": pygame.image.load("images/reina_negro.png").convert_alpha(),
            "r": pygame.image.load("images/torre_negro.png").convert_alpha(),
            "b": pygame.image.load("images/alfil_negro.png").convert_alpha(),
            "n": pygame.image.load("images/caballo_negro.png").convert_alpha()
        }

        self.rects_promocion_b = {
            "Q": pygame.Rect(172, 335, 100, 100),
            "R": pygame.Rect(281, 335, 100, 100),
            "B": pygame.Rect(392, 335, 100, 100),
            "N": pygame.Rect(501, 335, 100, 100)
        }

        self.rects_promocion_n = {
            "q": pygame.Rect(172, 335, 100, 100),
            "r": pygame.Rect(281, 335, 100, 100),
            "b": pygame.Rect(392, 335, 100, 100),
            "n": pygame.Rect(501, 335, 100, 100)
        }
        selec = None

        while not selec:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

                if event.type == pygame.MOUSEBUTTONDOWN and pieza.isupper():
                    for opcion, rect in self.rects_promocion_b.items():
                        if rect.collidepoint(event.pos):
                            selec = opcion
                            break

                if event.type == pygame.MOUSEBUTTONDOWN and pieza.islower():
                    for opcion, rect in self.rects_promocion_n.items():
                        if rect.collidepoint(event.pos):
                            selec = opcion
                            break

            pygame.draw.rect(self.screen, (160, 160, 160), (170, 335, 460, 130))
    
            colores = [(238, 238, 210), (118, 150, 86)]
            for i in range(4):
                pygame.draw.rect(self.screen, colores[i % 2], (180 + i * 110, 345, 110, 110))

            if pieza.isupper():
                for opcion, imagen in self.piezas_promocion_b.items():
                    rect = self.rects_promocion_b[opcion]
                    self.screen.blit(imagen, rect.topleft)
            else:
                for opcion, imagen in self.piezas_promocion_n.items():
                    rect = self.rects_promocion_n[opcion]
                    self.screen.blit(imagen, rect.topleft)

            pygame.display.update()

        if pieza.isupper():
            tupla = (selec, pygame.Rect(fila * ARISTA_IMG, columna * ARISTA_IMG, 100, 100))
        else:
            tupla = (selec, pygame.Rect(fila * ARISTA_IMG, columna * ARISTA_IMG, 100, 100))
        return selec, tupla

if __name__ == "__main__":
    main = Main()
    main.loop()