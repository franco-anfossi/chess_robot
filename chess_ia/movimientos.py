import chess

class Movimiento:
    def __init__(self):
        pass

    def verificar_movimiento(self, board, inicio, final):
        pos_inicial = chess.parse_square(inicio)
        if len(final) == 3: 
            pos_final = chess.parse_square(final[:-1])
            pieza_promocion = final[-1]
            promocion = {'q': chess.QUEEN, 'r': chess.ROOK, 
                        'b': chess.BISHOP, 'n': chess.KNIGHT}[pieza_promocion.lower()]
            
            move = chess.Move(pos_inicial, pos_final, promotion=promocion)
        else:
            pos_final = chess.parse_square(final)
            move = chess.Move(pos_inicial, pos_final)

        if board.is_legal(move):
            board.push(move)
            return True
        else:
            return False
    
    def verificar_jaque_mate(self, board):
        return board.is_checkmate()
    
    def verificar_tablas(self, board):
        comparacion_1 = board.is_stalemate() or board.is_insufficient_material()
        comparacion_2 = board.is_fivefold_repetition() or board.is_seventyfive_moves()
        return comparacion_1 or comparacion_2
    