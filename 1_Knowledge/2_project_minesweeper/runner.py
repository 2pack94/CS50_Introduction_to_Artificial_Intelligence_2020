import pygame
import sys

from minesweeper import Minesweeper, MinesweeperAI

HEIGHT = 8
WIDTH = 8
MINES = 8

# Colors
BLACK = (0, 0, 0)
GRAY = (180, 180, 180)
WHITE = (255, 255, 255)

# Create game
pygame.init()
size = width, height = 600, 400
screen = pygame.display.set_mode(size)

# Fonts
OPEN_SANS = "assets/fonts/OpenSans-Regular.ttf"
smallFont = pygame.font.Font(OPEN_SANS, 20)
mediumFont = pygame.font.Font(OPEN_SANS, 28)
largeFont = pygame.font.Font(OPEN_SANS, 40)

# Compute board size
BOARD_PADDING = 20
board_width = ((2 / 3) * width) - (BOARD_PADDING * 2)
board_height = height - (BOARD_PADDING * 2)
cell_size = int(min(board_width / WIDTH, board_height / HEIGHT))
board_origin = (BOARD_PADDING, BOARD_PADDING)

# Add images
flag = pygame.image.load("assets/images/flag.png")
flag = pygame.transform.scale(flag, (cell_size, cell_size))
mine = pygame.image.load("assets/images/mine.png")
mine = pygame.transform.scale(mine, (cell_size, cell_size))
mine_red = pygame.image.load("assets/images/mine_red.png")
mine_red = pygame.transform.scale(mine_red, (cell_size, cell_size))

# Mouse button number in pygame
MOUSE_BTN_LEFT = 1
MOUSE_BTN_RIGHT = 3

# Create game and AI agent
game = Minesweeper(height=HEIGHT, width=WIDTH, mines=MINES)
ai = MinesweeperAI(height=HEIGHT, width=WIDTH)

# Keep track of flagged cells (a flag is set on a right-click)
flags = set()
# Set to True if a mine was hit
lost = False
# Set to True if all cells were revealed
won = False
# Store position of the mine when clicking on it to draw that mine in red.
mine_clicked = (0, 0)

# Show instructions initially
instructions = True

while True:
    mouse_btn, mouse_pos = None, None

    # process all events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            sys.exit()
        elif event.type == pygame.MOUSEBUTTONDOWN:
            mouse_pos = event.pos
            mouse_btn = event.button

    screen.fill(BLACK)

    # Show game instructions
    if instructions:
        # Title
        title = largeFont.render("Play Minesweeper", True, WHITE)
        titleRect = title.get_rect()
        titleRect.center = ((width / 2), 50)
        screen.blit(title, titleRect)

        # Rules
        rules = [
            "Click a cell to reveal it.",
            "Right-click a cell to mark it as a mine.",
            "Reveal all cells that are not mines to win!"
        ]
        for i, rule in enumerate(rules):
            line = smallFont.render(rule, True, WHITE)
            lineRect = line.get_rect()
            lineRect.center = ((width / 2), 150 + 30 * i)
            screen.blit(line, lineRect)

        # Play game button
        buttonRect = pygame.Rect((width / 4), (3 / 4) * height, width / 2, 50)
        buttonText = mediumFont.render("Play Game", True, BLACK)
        buttonTextRect = buttonText.get_rect()
        buttonTextRect.center = buttonRect.center
        pygame.draw.rect(screen, WHITE, buttonRect)
        screen.blit(buttonText, buttonTextRect)

        # Check if play button clicked
        if mouse_btn == MOUSE_BTN_LEFT:
            if buttonRect.collidepoint(mouse_pos):
                instructions = False

        pygame.display.flip()
        continue

    # Draw board
    cells = []
    for i in range(HEIGHT):
        row = []
        for j in range(WIDTH):

            # Draw rectangle for cell
            rect = pygame.Rect(
                board_origin[0] + j * cell_size,
                board_origin[1] + i * cell_size,
                cell_size, cell_size
            )
            pygame.draw.rect(screen, GRAY, rect)
            pygame.draw.rect(screen, WHITE, rect, 3)

            # draw a mine, flag, or number
            if game.isMine((i, j)) and lost:
                if (i, j) == mine_clicked:
                    screen.blit(mine_red, rect)
                else:
                    screen.blit(mine, rect)
            elif (i, j) in flags:
                screen.blit(flag, rect)
            elif game.board_revealed[i][j] is not None:
                neighbors = smallFont.render(
                    str(game.board_revealed[i][j]),
                    True, BLACK
                )
                neighborsTextRect = neighbors.get_rect()
                neighborsTextRect.center = rect.center
                screen.blit(neighbors, neighborsTextRect)

            row.append(rect)
        cells.append(row)

    # AI Move button
    aiButton = pygame.Rect(
        (2 / 3) * width + BOARD_PADDING, (1 / 3) * height - 50,
        (width / 3) - BOARD_PADDING * 2, 50
    )
    buttonText = mediumFont.render("AI Move", True, BLACK)
    buttonRect = buttonText.get_rect()
    buttonRect.center = aiButton.center
    pygame.draw.rect(screen, WHITE, aiButton)
    screen.blit(buttonText, buttonRect)

    # Reset button
    resetButton = pygame.Rect(
        (2 / 3) * width + BOARD_PADDING, (1 / 3) * height + 20,
        (width / 3) - BOARD_PADDING * 2, 50
    )
    buttonText = mediumFont.render("Reset", True, BLACK)
    buttonRect = buttonText.get_rect()
    buttonRect.center = resetButton.center
    pygame.draw.rect(screen, WHITE, resetButton)
    screen.blit(buttonText, buttonRect)

    # Display text
    text = "Lost" if lost else "Won" if won else ""
    text = mediumFont.render(text, True, WHITE)
    textRect = text.get_rect()
    textRect.center = ((5 / 6) * width, (2 / 3) * height)
    screen.blit(text, textRect)

    move = None

    # Check for a right-click to toggle flagging
    if mouse_btn == MOUSE_BTN_RIGHT and not lost and not won:
        for i in range(HEIGHT):
            for j in range(WIDTH):
                if cells[i][j].collidepoint(mouse_pos) and game.board_revealed[i][j] is None:
                    if (i, j) in flags:
                        flags.remove((i, j))
                    else:
                        flags.add((i, j))

    elif mouse_btn == MOUSE_BTN_LEFT:
        # If AI button clicked, make an AI move
        if aiButton.collidepoint(mouse_pos) and not lost and not won:
            move = ai.makeSafeMove()
            if move is None:
                move = ai.makeRandomMove()
                if move is None:
                    print("No moves left to make.")
                else:
                    print("No known safe moves, AI making random move.")
            else:
                print("AI making safe move.")
            if move and move in flags:
                flags.remove(move)

        # Reset game state
        elif resetButton.collidepoint(mouse_pos):
            game = Minesweeper(height=HEIGHT, width=WIDTH, mines=MINES)
            ai = MinesweeperAI(height=HEIGHT, width=WIDTH)
            flags = set()
            lost = False
            won = False
            continue

        # User-made move
        elif not lost and not won:
            for i in range(HEIGHT):
                for j in range(WIDTH):
                    if (cells[i][j].collidepoint(mouse_pos)
                            and (i, j) not in flags
                            and game.board_revealed[i][j]) is None:
                        move = (i, j)

    # Make move and update AI knowledge
    if move:
        if game.isMine(move):
            lost = True
            mine_clicked = move
        else:
            game.reveal(move)
            count = game.board_revealed[move[0]][move[1]]
            ai.addKnowledge(move, count)
            if game.isWon():
                won = True

    pygame.display.flip()
