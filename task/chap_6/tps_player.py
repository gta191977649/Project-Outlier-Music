import pygame
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.backends.backend_agg as agg
import time
from feature.pattern import extractTontalPitchDistancePattern
from feature.pattern import extractChromaticPattern
from model.song import Song

# Load song data
#H5_PATH = "/Users/nurupo/Desktop/dev/music4all/custom/MV君はメロディー Short ver.  AKB48[公式].h5"
AUDIO_PATH = "/Users/nurupo/Desktop/dev/music4all/mp3/Europe - The Final Countdown (Official Video).mp3"
#song = Song.from_h5(H5_PATH)
song = Song(id="AKB",title="TEST", artist="TEST",file=AUDIO_PATH)
chords = [(float(item[0]), item[1], item[2]) for item in song.chord]
key = f"{song.key}:{song.mode[:3]}"
print(f"key is {key}")
chord_labels = [item[2] for item in chords]
tps = extractTontalPitchDistancePattern(chord_labels, mode="profile", key=key)
#tps = extractChromaticPattern(chord_labels)

# Initialize Pygame
pygame.init()
window = pygame.display.set_mode((1100, 200), pygame.DOUBLEBUF)
pygame.display.set_caption('Tonal Pitch Sequence Visualization')

# Load fonts
font = pygame.font.Font(None, 36)  # Default font and size

# Matplotlib figure setup
fig = plt.figure(figsize=(11, 1.5), dpi=100)
ax = fig.add_subplot(111)
line, = ax.step(range(len(tps)), tps, where='mid',color="b")
vline = ax.axvline(x=0, color='r', linestyle='--')
ax.set_xlim(0, len(tps))
ax.set_ylim(min(tps), max(tps))
canvas = agg.FigureCanvasAgg(fig)

# Audio setup
pygame.mixer.init()
sound = pygame.mixer.Sound(AUDIO_PATH)

# Helper function to draw buttons and text
def draw_buttons_and_text(playback_status, current_time):
    play_text = 'Pause' if playback_status else 'Play'
    text_surface = font.render(f"Time: {current_time:.2f}s - {play_text}", True, (255, 255, 255))
    window.blit(text_surface, (10, 170))  # Adjusted placement

    # Draw buttons lower than the text
    pygame.draw.rect(window, (0, 255, 0), (600, 170, 60, 30))  # Play/Pause
    pygame.draw.rect(window, (255, 0, 0), (670, 170, 60, 30))  # Stop
    pygame.draw.rect(window, (0, 0, 255), (740, 170, 60, 30))  # Reset

# Control flags and initial times
running = True
playback_status = False
current_time = 0.0
start_time = 0.0
paused_time = 0.0

# Main loop
# Main loop
while running:
    window.fill((0, 0, 0))  # Clear screen with black

    # Handle events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            mouse_x, mouse_y = pygame.mouse.get_pos()
            # Play/Pause button
            if 600 <= mouse_x <= 660 and 170 <= mouse_y <= 200:  # Check coordinates for Play/Pause
                playback_status = not playback_status
                if playback_status:
                    sound.play()
                    start_time = time.time() - paused_time
                else:
                    paused_time = time.time() - start_time
                    sound.stop()
            # Stop button
            elif 670 <= mouse_x <= 730 and 170 <= mouse_y <= 200:  # Check coordinates for Stop
                sound.stop()
                playback_status = False
                current_time = 0
                paused_time = 0
                start_time = 0  # Reset start time
            # Reset button
            elif 740 <= mouse_x <= 800 and 170 <= mouse_y <= 200:  # Check coordinates for Reset
                sound.stop()
                sound.play()
                start_time = time.time()
                playback_status = True
                current_time = 0
                paused_time = 0

    # Update and draw only if playback is active
    if playback_status:
        current_time = time.time() - start_time
        x_position = next((i for i, item in enumerate(chords) if item[0] > current_time), -1)
        if x_position == -1:
            x_position = len(tps)
        vline.set_xdata([x_position, x_position])
        # Render the plot to a buffer and convert to a Pygame surface
        canvas.draw()
        renderer = canvas.get_renderer()
        raw_data = renderer.tostring_rgb()
        size = canvas.get_width_height()
        surf = pygame.image.fromstring(raw_data, size, "RGB")
        window.blit(surf, (0, 0))
    else:
        if not paused_time:  # If stopped, reset the view
            vline.set_xdata([0, 0])  # Reset vertical line to start
            canvas.draw()
            renderer = canvas.get_renderer()
            raw_data = renderer.tostring_rgb()
            size = canvas.get_width_height()
            surf = pygame.image.fromstring(raw_data, size, "RGB")
            window.blit(surf, (0, 0))

    draw_buttons_and_text(playback_status, current_time)
    pygame.display.flip()
    pygame.time.wait(10)  # Sleep briefly to reduce CPU usage

pygame.quit()
