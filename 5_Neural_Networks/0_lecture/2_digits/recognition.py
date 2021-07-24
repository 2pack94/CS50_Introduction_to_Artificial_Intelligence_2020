import numpy as np
import pygame
import sys
import tensorflow as tf

# Load a previously trained model (see handwriting.py) and use it to classify handwritten digits.

# Check command-line arguments
if len(sys.argv) != 2:
    sys.exit("Usage: python recognition.py model")
model = tf.keras.models.load_model(sys.argv[1])

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

# Mouse button number in pygame
MOUSE_BTN_LEFT = 1
MOUSE_BTN_RIGHT = 3

# Start pygame
pygame.init()
size = width, height = 600, 400
screen = pygame.display.set_mode(size)

# Fonts
OPEN_SANS = "assets/fonts/OpenSans-Regular.ttf"
smallFont = pygame.font.Font(OPEN_SANS, 20)
largeFont = pygame.font.Font(OPEN_SANS, 40)

ROWS, COLS = 28, 28

OFFSET = 20
CELL_SIZE = 10

handwriting = [[0] * COLS for _ in range(ROWS)]
prediction = None

while True:

    mouse_btn, mouse_pos = None, None

    # process all events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            sys.exit()
        elif event.type == pygame.MOUSEBUTTONDOWN:
            mouse_pos = event.pos
            mouse_btn = event.button

    mouse_1_pressed, _, _ = pygame.mouse.get_pressed()
    if mouse_1_pressed:
        mouse_pressed_pos = pygame.mouse.get_pos()

    screen.fill(BLACK)

    # Draw each grid cell
    cells = []
    for i in range(ROWS):
        row = []
        for j in range(COLS):
            rect = pygame.Rect(
                OFFSET + j * CELL_SIZE,
                OFFSET + i * CELL_SIZE,
                CELL_SIZE, CELL_SIZE
            )

            # If cell has been written on, darken cell
            if handwriting[i][j]:
                channel = 255 - (handwriting[i][j] * 255)
                pygame.draw.rect(screen, (channel, channel, channel), rect)

            # Draw blank cell
            else:
                pygame.draw.rect(screen, WHITE, rect)
            pygame.draw.rect(screen, BLACK, rect, 1)

            # If writing on this cell, fill in current cell and neighbors
            if mouse_1_pressed and rect.collidepoint(mouse_pressed_pos):
                handwriting[i][j] = 250 / 255
                if i + 1 < ROWS:
                    handwriting[i + 1][j] = 220 / 255
                if j + 1 < COLS:
                    handwriting[i][j + 1] = 220 / 255
                if i + 1 < ROWS and j + 1 < COLS:
                    handwriting[i + 1][j + 1] = 190 / 255

    # Reset button
    resetButton = pygame.Rect(
        30, OFFSET + ROWS * CELL_SIZE + 30,
        100, 30
    )
    resetText = smallFont.render("Reset", True, BLACK)
    resetTextRect = resetText.get_rect()
    resetTextRect.center = resetButton.center
    pygame.draw.rect(screen, WHITE, resetButton)
    screen.blit(resetText, resetTextRect)

    # Classify button
    classifyButton = pygame.Rect(
        150, OFFSET + ROWS * CELL_SIZE + 30,
        100, 30
    )
    classifyText = smallFont.render("Classify", True, BLACK)
    classifyTextRect = classifyText.get_rect()
    classifyTextRect.center = classifyButton.center
    pygame.draw.rect(screen, WHITE, classifyButton)
    screen.blit(classifyText, classifyTextRect)

    # Reset drawing
    if mouse_btn == MOUSE_BTN_LEFT and resetButton.collidepoint(mouse_pos):
        handwriting = [[0] * COLS for _ in range(ROWS)]
        prediction = None

    # Generate prediction
    if mouse_btn == MOUSE_BTN_LEFT and classifyButton.collidepoint(mouse_pos):
        prediction = model.predict(
            [np.array(handwriting).reshape(1, 28, 28, 1)]
        )

    # Show prediction
    if prediction is not None:
        classification = prediction.argmax()
        classificationText = largeFont.render(str(classification), True, WHITE)
        classificationRect = classificationText.get_rect()
        grid_size = OFFSET * 2 + CELL_SIZE * COLS
        classificationRect.center = (
            grid_size + ((width - grid_size) / 2),
            50
        )
        screen.blit(classificationText, classificationRect)

        for i in range(len(prediction[0])):
            probText = smallFont.render(f"{i}: {prediction[0][i]:.2e}", True, WHITE)
            probRect = probText.get_rect()
            probRect.topleft = (
                grid_size + 20,
                100 + 25 * i
            )
            screen.blit(probText, probRect)

    pygame.display.flip()
