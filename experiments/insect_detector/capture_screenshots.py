
import pyautogui
import time
from pathlib import Path

OUTPUT = Path(
    r"D:\InsectClass\experiments\insect_detector\data\negative_extra"
)

OUTPUT.mkdir(parents=True, exist_ok=True)

print("INSECTCLASS AI - Screenshot Collector")
print("Open different windows without insect images.")
print("Screenshots will begin in 5 seconds.")
print("Press Ctrl+C to stop early.")

time.sleep(5)

for i in range(20):
    image = pyautogui.screenshot()

    filename = OUTPUT / f"screenshot_{i:03d}.png"
    image.save(filename)

    print(f"Saved screenshot {i + 1}/20")

    if i < 19:
        time.sleep(5)

print("\n20 screenshots captured successfully!")
