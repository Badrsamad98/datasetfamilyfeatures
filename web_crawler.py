import requests
import re
import json
import time
import os
import logging
from pathlib import Path
import csv
from requests.exceptions import HTTPError

'''Constantes'''
URL = 'https://duckduckgo.com/'
REQUEST_URL = URL + "i.js"
HEADERS = {
    'authority': 'duckduckgo.com',
    'accept': 'application/json, text/javascript, */*; q=0.01',
    'sec-fetch-dest': 'empty',
    'x-requested-with': 'XMLHttpRequest',
    'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/80.0.3987.163 Safari/537.36',
    'sec-fetch-site': 'same-origin',
    'sec-fetch-mode': 'cors',
    'referer': 'https://duckduckgo.com/',
    'accept-language': 'en-US,en;q=0.9',
}

'''Logging'''
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

'''Function that launches the script by reading the input and config files'''


def launch_script():
    '''Configuration'''
    configFile = open('config.txt', 'r')
    config = configFile.readlines()
    height = int(config[0].rstrip())
    width = int(config[1].rstrip())
    max_results = int(config[2].rstrip())

    # Reading the input file that contains the names to search
    inputFile = open('input.txt', 'r')
    lines = inputFile.readlines()

    family_index = 0
    family = []

    for line in lines:
        if line.rstrip() == 'begin':  # End code word that indicates that we have retrieved an entire familt
            if(family_index != 0):
                search_for_family_images(
                    family, family_index, max_results, height, width)

            # Reset the variables and start create the folder for a new family
            family = []
            family_index += 1
            Path(os.getcwd() + '/downoalds' + '/Family_' +
                 str(family_index)).mkdir(parents=True, exist_ok=True)

        else:
            family.append(line)


'''Searches for the images of a family composed of three family members, saves the images and the csv identification files'''


def search_for_family_images(family, family_index, max_results=None, height=800, width=600):
    global REQUEST_URL
    for index, keywords in enumerate(family):

        searchObj = get_token(keywords)
        folder_path = create_images_folder_by_index(
            index, keywords, family_index)

        params = (
            ('l', 'us-en'),
            ('o', 'json'),
            ('q', keywords),
            ('vqd', searchObj.group(1)),
            ('f', ',,,'),
            ('p', '1'),
            ('v7exp', 'a'),
        )

        logger.debug("Hitting Url : %s", REQUEST_URL)

        number_results = 0
        imagelinks = []

        while True:
            while True:
                try:
                    res = requests.get(
                        REQUEST_URL, headers=HEADERS, params=params)
                    data = json.loads(res.text)
                    break
                except ValueError:
                    logger.debug(
                        "Hitting Url Failure - Sleep and Retry: %s", REQUEST_URL)
                    time.sleep(1)
                    continue

            logger.debug("Hitting Url Success : %s", REQUEST_URL)

            for k in data["results"]:
                if(number_results >= max_results):
                    break
                if(k['height'] > height and k['width'] > width):
                    imagelinks.append(k['image'])
                    number_results += 1

            print(f'found {len(imagelinks)} images')
            print('Start downloading...')

            get_images_by_images_links(
                family, index, imagelinks, folder_path, keywords)

            print('Done')

            if(number_results >= max_results):
                break
            # find the next page with more pictures
            if "next" not in data:
                logger.debug("No Next Page - Exiting")
                break

            REQUEST_URL = URL + data["next"]


'''Get token from search engine'''


def get_token(keywords):
    params = {
        'q': keywords
    }

    logger.debug("Hitting DuckDuckGo for Token")

    #   First make a request to above URL, and parse out the 'vqd'
    #   This is a special token, which should be used in the subsequent request
    res = requests.post(URL, data=params)
    searchObj = re.search(r'vqd=([\d-]+)\&', res.text, re.M | re.I)

    if not searchObj:
        logger.error("Token Parsing Failed !")
        return -1

    logger.debug("Obtained Token")
    return searchObj


'''Creating the folder where to store the pictures of each individual of a family'''


def create_images_folder_by_index(index, keywords, familyindex):
    # The folders containing the images
    SAVE_FOLDER = os.getcwd() + '/downoalds/' + '/Family_' + str(familyindex) + '/'

    switcher = {
        0: SAVE_FOLDER + file_name_formatter(keywords)+'_father',
        1: SAVE_FOLDER + file_name_formatter(keywords)+'_mother',
        2: SAVE_FOLDER + file_name_formatter(keywords)+'_daughter',
    }

    try:
        folder_path = switcher[index]
    except KeyError:
        logger.error("Invalid index")
        return -1

    Path(folder_path).mkdir(parents=True, exist_ok=True)
    return folder_path


'''Formatting the name of the file'''


def file_name_formatter(keywords):
    words = keywords.rstrip().split(" ")
    result = ""
    for word in words:
        result += word + "_"
    result = result[:-1]
    return result


def get_images_by_images_links(family, index, imagelinks, folder_path, keywords):
    for i, imagelink in enumerate(imagelinks):
        # open image link and save as file
        while True:
            try:
                response = requests.get(imagelink)
                # If the response was successful, no Exception will be raised
                response.raise_for_status()
            except HTTPError as http_err:
                print(f'HTTP error occurred: {http_err}')  # Python 3.6
            except Exception as err:
                print(f'Other error occurred: {err}')  # Python 3.6
            else:
                print('Success!')
            break

        save_image(folder_path, keywords, i, response)

        create_csv_identification_file(
            family, folder_path, index, i, keywords, imagelink)


'''Save the image'''


def save_image(folder_path, keywords, i, response):
    imagename = Path(folder_path + '/' +
                     file_name_formatter(keywords) + str(i+1) + '.jpg')
    with open(imagename, 'wb') as file:
        file.write(response.content)


'''Create the CSV File that identifies each image'''


def create_csv_identification_file(family, folder_path, index, i, keywords, imagelink):
    linkname = Path(folder_path + '/' +
                    file_name_formatter(keywords) + str(i+1) + '.csv')

    with open(linkname, 'w', newline='') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=';',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        spamwriter.writerow(
            ['Filename', file_name_formatter(keywords) + str(i+1) + '.jpg'])
        spamwriter.writerow(['Name', keywords.rstrip()])
        spamwriter.writerow(['Father', family[0].rstrip()])
        spamwriter.writerow(['Mother', family[1].rstrip()])
        spamwriter.writerow(['Daughter', family[2].rstrip()])
        spamwriter.writerow(['Kin relation', 'Father' if index == 0 else (
            'Mother' if index == 1 else 'Daughter')])
        spamwriter.writerow(['Gender', 'Male' if index == 0 else 'Female'])
        spamwriter.writerow(['URL', imagelink])


'''Execute the script'''
launch_script()
