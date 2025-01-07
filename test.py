from selenium import webdriver
from selenium.webdriver.common.by import By
import time
import pyautogui
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.firefox.service import Service as FirefoxService
from selenium.webdriver.firefox.options import Options

# options = Options()
# options.add_argument("-headless")

driver = webdriver.Firefox()
# driver.minimize_window()
driver.get('https://www.dzine.ai/')

#button = driver.find_element(By.XPATH, "//button[@class='closeBtn']")
#button.click()

time.sleep(5)

button = driver.find_element(By.XPATH, "/html/body/div[1]/div/div[1]/div/div[2]/div/div/div/div[2]/button")
button.click()

log = driver.find_element(By.XPATH, '/html/body/div[1]/div/div[1]/div/div[2]/div/div/div/div[2]/div[2]/span')
log.click()

button = driver.find_element(By.XPATH, '/html/body/div[3]/div/div/div[1]/button[2]')
button.click()

email = "xiaotest57@gmail.com"
pwd = "cartoonify"
username = driver.find_element(By.XPATH, "//input[@id='username']")
username.send_keys(email)
password = driver.find_element(By.XPATH, "//input[@id='password']")
password.send_keys(pwd)
button = driver.find_element(By.XPATH, "//button[@class='btn primary big continue']")
button.click()
time.sleep(5)
button = driver.find_element(By.XPATH, "//button[@class='project-item create']")
button.click()
time.sleep(5)

button = driver.find_element(By.XPATH, "/html/body/div[1]/div/div/div[4]/div/div[4]/button[1]")
button.click()

# button = driver.find_element(By.XPATH, "/html/body/div[1]/div/div/div[3]/div[7]/div/span")
# button.click()

button = driver.find_element(By.XPATH, "//div[@class='top circle1']")
button.click()
time.sleep(2)

pyautogui.write("/home/xiaoyo/ours/input/photo.png")
pyautogui.press('enter')

button = driver.find_element(By.XPATH, "//div[@id='img2imgBtn']")
button.click()
time.sleep(20)

button = driver.find_element(By.XPATH, "//button[@id='img2img-style-btn']")
button.click()

time.sleep(8)

element = driver.find_element(By.XPATH, "/html/body/div[5]/div/div[1]/div[2]/div[1]/div/div[2]/span")
element.click()

time.sleep(2)

element = driver.find_element(By.XPATH, "/html/body/div[5]/div/div[1]/div[2]/div[2]/div[3]/div[1]/div[2]/ul/li/div")
print(element.is_displayed())
element.click()


# button = driver.find_element(By.XPATH, "/html/body/div[4]/div/div[1]/div[2]/div[2]/div[3]/div[1]/div[2]/ul/li[73]/div/div/button")
# button.click()

button = driver.find_element(By.XPATH, "//button[@id='img2img-generate-btn']")
button.click()

time.sleep(30)
# button = driver.find_element(By.XPATH, "//span[@class='close-btn']")
# if button:
#     button.click()

time.sleep(50)
element = driver.find_element(By.XPATH, "/html/body/div[1]/div/div/div[3]/div[2]/div[2]/div/div[2]/div[1]/div/div/div[2]/button[1]/div[2]")

actions = ActionChains(driver)
actions.move_to_element(element)
actions.perform()

button = driver.find_element(By.XPATH, "/html/body/div[1]/div/div/div[3]/div[2]/div[2]/div/div[2]/div[1]/div/div/div[2]/button[1]/div[2]/div/div[2]/button")
button.click()

button = driver.find_element(By.XPATH, '/html/body/div[7]/div/div/div[3]/button')
button.click()

time.sleep(3)
driver.close()
