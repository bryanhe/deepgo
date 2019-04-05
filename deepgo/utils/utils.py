import torch

def to_coord(x):
    assert(len(x) == 2)
    return ord(x[0]) - ord("a"), ord(x[1]) - ord("a")


def capture(board, player):
    liberties = has_liberties(board, torch.as_tensor([player], dtype=torch.long))

    # print(board2str(board) + "\n")
    if len(board.shape) == 3:
        board[player, :, :][~liberties] = 0
    else:
        board[:, player, :, :][~liberties] = 0
    # print(board2str(board) + "\n\n")

# @torch.jit.script
def _has_liberties(board, player):
    # Liberties is either an empty spot, or is a piece with a liberty
    liberties = ((board[:, 0, :, :] == 0) & (board[:, 1, :, :] == 0))
    prev = liberties.clone()
    first = True
    while first or bool((liberties != prev).any()):
        first = False
        prev = liberties.clone()
        liberties[:, :18, :] |= (liberties[:,  1:, :] & board[:, player[0], :18, :])
        liberties[:, 1:,  :] |= (liberties[:, :18, :] & board[:, player[0],  1:, :])
        liberties[:, :, :18] |= (liberties[:, :,  1:] & board[:, player[0], :, :18])
        liberties[:, :,  1:] |= (liberties[:, :, :18] & board[:, player[0], :,  1:])
    return liberties

def has_liberties(board, player):
    if len(board.shape) == 3:
        assert(board.shape == torch.Size([2, 19, 19]))
        board = board.view(1, 2, 19, 19)
        remove_dim = True
    else:
        assert(len(board.shape) == 4)
        assert(board.shape[1] == 2)
        assert(board.shape[2] == 19)
        assert(board.shape[3] == 19)
        remove_dim = False

    liberties = _has_liberties(board, torch.as_tensor([player], dtype=torch.long))

    if remove_dim:
        liberties.squeeze_(0)

    return liberties


def board2str(board):
    assert(board.shape == torch.Size([2, 19, 19]))
    res = ""
    for i in range(19):
        for j in range(19):
            if board[0, i, j] != 0:
                res += "X"
            elif board[1, i, j] != 0:
                res += "O"
            else:
                res += "."
        res += "\n"
    return res
