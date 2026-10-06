"""
Play White Move Demo Script
===========================
Automates playing as White against AlphaGo Zero on http://127.0.0.1:9000
and saves a full-resolution screenshot proving the White stone is placed on the board.
"""

import time
import os
import shutil
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options as ChromeOptions
from selenium.webdriver.edge.options import Options as EdgeOptions


def click_element(driver, element):
    driver.execute_script("arguments[0].dispatchEvent(new Event('click', {bubbles: true}));", element)


def main():
    target_url = "http://127.0.0.1:9000"
    print(f"Opening local server at {target_url}...")

    options = ChromeOptions()
    options.add_argument("--window-size=1400,950")
    options.add_argument("--no-sandbox")

    driver = None
    try:
        driver = webdriver.Chrome(options=options)
    except Exception:
        edge_opts = EdgeOptions()
        edge_opts.add_argument("--window-size=1400,950")
        driver = webdriver.Edge(options=edge_opts)

    driver.implicitly_wait(10)
    wait = WebDriverWait(driver, 30)

    try:
        driver.get(target_url)
        time.sleep(2)

        # Select White
        print("Selecting White player...")
        white_btn = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "[data-color='-1']")))
        click_element(driver, white_btn)
        time.sleep(0.5)

        # Click Start New Game
        print("Starting new game as White...")
        new_game_btn = driver.find_element(By.ID, "new-game")
        click_element(driver, new_game_btn)
        time.sleep(2)

        # Wait for AI's Black opening move to complete
        print("Waiting for Black AI to play Move #1...")
        try:
            wait.until(EC.invisibility_of_element_located((By.ID, "board-thinking")))
        except Exception:
            pass

        time.sleep(1)

        # Find empty intersection J4 / I4 (row 3, col 8) for White's move
        intersection = wait.until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "g.intersection[data-row='3'][data-col='8']"))
        )
        print("Clicking empty intersection (row 3, col 8) to place White stone...")
        click_element(driver, intersection)

        # Wait for White move to process
        time.sleep(2)
        try:
            wait.until(EC.invisibility_of_element_located((By.ID, "board-thinking")))
        except Exception:
            pass

        time.sleep(1)

        # Take proof screenshot
        screen_path = os.path.abspath("white_stone_placed_proof.png")
        driver.save_screenshot(screen_path)
        print(f"Proof screenshot saved to: {screen_path}")

        # Copy to artifact directory for display
        artifact_dir = r"C:\Users\CHANDRAKATH REDDY\.gemini\antigravity-ide\brain\d28f710b-e321-48bc-aca2-1e1a70519778"
        os.makedirs(artifact_dir, exist_ok=True)
        artifact_img = os.path.join(artifact_dir, "white_stone_placed_proof.png")
        shutil.copy(screen_path, artifact_img)
        print(f"Copied proof screenshot to artifact dir: {artifact_img}")

    finally:
        if driver:
            driver.quit()

if __name__ == "__main__":
    main()
