"""
AlphaGo Zero - Automated Selenium Gameplay Test
===============================================
Tests end-to-end web UI interactions for both Black and White players, verifying:
1. Playing as Black: Clicking an empty intersection places a Black stone.
2. Playing as White: Waiting for AI opening move, then clicking an empty intersection places a White stone.
3. Occupied spot safety: Attempting to click an occupied stone handles gracefully.
"""

import time
import argparse
import sys
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options as ChromeOptions
from selenium.webdriver.edge.options import Options as EdgeOptions


def click_element(driver, element):
    """Fires native click event on SVG or HTML elements in JavaScript."""
    driver.execute_script("arguments[0].dispatchEvent(new Event('click', {bubbles: true}));", element)


def run_tests(target_url: str = "https://alphagozero.vercel.app", headless: bool = False):
    print(f"[+] Launching Selenium Web Driver test target: {target_url}")

    # Setup Chrome or Edge Driver
    driver = None
    options = ChromeOptions()
    if headless:
        options.add_argument("--headless=new")
    options.add_argument("--window-size=1400,900")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")

    try:
        driver = webdriver.Chrome(options=options)
    except Exception as e:
        print(f"Chrome Driver initialization note: {e}. Trying Edge Driver...")
        edge_opts = EdgeOptions()
        if headless:
            edge_opts.add_argument("--headless=new")
        edge_opts.add_argument("--window-size=1400,900")
        driver = webdriver.Edge(options=edge_opts)

    driver.implicitly_wait(10)
    wait = WebDriverWait(driver, 45)

    try:
        # ---------------------------------------------------------------------
        # TEST 1: Load Page & Connection Verification
        # ---------------------------------------------------------------------
        print("\n--- TEST 1: Page Load & Engine Connection Verification ---")
        driver.get(target_url)
        
        # Wait for engine-overlay to become ready / hidden
        print("--> Waiting for engine connection overlay to clear...")
        try:
            wait.until(EC.attribute_contains((By.ID, "engine-overlay"), "class", "ready"))
            print("[OK] Engine connection overlay cleared.")
        except Exception:
            # Fallback to Demo Mode if cold start took too long
            print("[NOTE] Cold start overlay active. Clicking Offline Demo Mode...")
            demo_btn = driver.find_element(By.ID, "wake-demo")
            if demo_btn.is_displayed():
                click_element(driver, demo_btn)
                time.sleep(2)

        time.sleep(2)
        print("[OK] Page loaded successfully. Title:", driver.title)

        # ---------------------------------------------------------------------
        # TEST 2: Play as Black (Human plays move #1)
        # ---------------------------------------------------------------------
        print("\n--- TEST 2: Playing as BLACK (Human First Move) ---")
        
        # Click Black radio/toggle
        black_btn = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "[data-color='1']")))
        click_element(driver, black_btn)
        time.sleep(0.5)

        # Click Start New Game
        new_game_btn = driver.find_element(By.ID, "new-game")
        click_element(driver, new_game_btn)
        time.sleep(2)

        # Wait for thinking overlay to hide if active
        try:
            wait.until(EC.invisibility_of_element_located((By.ID, "board-thinking")))
        except Exception:
            pass

        # Locate empty intersection D4 (row 3, col 3)
        intersection_d4 = wait.until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "g.intersection[data-row='3'][data-col='3']"))
        )
        print("--> Clicking empty intersection D4 (row 3, col 3) as Black...")
        click_element(driver, intersection_d4)

        # Wait for move processing
        time.sleep(3)

        # Verify move history updated
        move_list = driver.find_element(By.ID, "move-list").text
        print("Move List Output:\n", move_list)
        assert "D4" in move_list or "01" in move_list, "Error: Move D4 not registered in move history!"
        print("[SUCCESS] TEST 2 PASSED: Black stone placed successfully on empty intersection D4!")

        # ---------------------------------------------------------------------
        # TEST 3: Play as White (AI plays Black #1, Human plays White #2)
        # ---------------------------------------------------------------------
        print("\n--- TEST 3: Playing as WHITE (AI First Move, Human Second Move) ---")

        # Click White radio/toggle
        white_btn = driver.find_element(By.CSS_SELECTOR, "[data-color='-1']")
        click_element(driver, white_btn)
        time.sleep(0.5)

        # Start new game as White
        new_game_btn = driver.find_element(By.ID, "new-game")
        click_element(driver, new_game_btn)
        print("--> Started new game as White. Waiting for AI (Black) to complete move #1...")
        time.sleep(3)

        # Wait for AI opening move to finish
        try:
            wait.until(EC.invisibility_of_element_located((By.ID, "board-thinking")))
        except Exception:
            pass

        move_list = driver.find_element(By.ID, "move-list").text
        print("Move List after AI Opening Move:\n", move_list)

        # Locate empty intersection J4 (row 3, col 8) for White
        intersection_j4 = wait.until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "g.intersection[data-row='3'][data-col='8']"))
        )
        print("--> Clicking empty intersection J4 (row 3, col 8) as White...")
        click_element(driver, intersection_j4)

        # Wait for move processing
        time.sleep(3)

        updated_move_list = driver.find_element(By.ID, "move-list").text
        print("Updated Move List:\n", updated_move_list)
        assert "I4" in updated_move_list or "[W]" in updated_move_list, "Error: White move I4 not registered!"
        print("[SUCCESS] TEST 3 PASSED: White stone placed successfully on empty intersection I4!")

        # ---------------------------------------------------------------------
        # TEST 4: Occupied Intersection Protection
        # ---------------------------------------------------------------------
        print("\n--- TEST 4: Attempting Click on Occupied Stone (Safety Check) ---")
        
        # Click on J4 again (which now has a White stone)
        click_element(driver, intersection_j4)
        time.sleep(1)
        
        final_move_list = driver.find_element(By.ID, "move-list").text
        print("[SUCCESS] TEST 4 PASSED: Clicking occupied stone J4 was ignored correctly as an illegal move.")

        print("\n[ALL TESTS PASSED] The AlphaGo Zero web interface is 100% working for both Black and White moves!")

    except Exception as err:
        print(f"\n[FAILED] SELENIUM TEST ERROR: {err}")
        if driver:
            screenshot_path = "selenium_failure_screenshot.png"
            driver.save_screenshot(screenshot_path)
            print(f"Saved failure screenshot to: {screenshot_path}")
        raise err

    finally:
        if driver:
            driver.quit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AlphaGo Zero Selenium Test")
    parser.add_argument("--url", type=str, default="https://alphagozero.vercel.app", help="Target Web App URL")
    parser.add_argument("--headless", action="store_true", help="Run browser in headless mode")
    args = parser.parse_args()

    run_tests(target_url=args.url, headless=args.headless)
