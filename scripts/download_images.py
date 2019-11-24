from google_images_download import google_images_download

response = google_images_download.googleimagesdownload()
arguments = {"keywords": "weapon", "limit": 200, "print_urls": False, "no_directory": True, "chromedriver":
    "C:\\chromedriver\\chromedriver.exe", "output_directory": "weapons"}
response.download(arguments)

response = google_images_download.googleimagesdownload()
arguments = {"keywords": "nudity", "limit": 1000, "print_urls": False, "no_directory": True, "chromedriver":
    "C:\\chromedriver\\chromedriver.exe", "output_directory": "nudity"}
response.download(arguments)

response = google_images_download.googleimagesdownload()
arguments = {"keywords": "vehicles cars", "limit": 200, "print_urls": False, "no_directory": True, "chromedriver":
    "C:\\chromedriver\\chromedriver.exe", "output_directory": "cars"}
response.download(arguments)

response = google_images_download.googleimagesdownload()
arguments = {"keywords": "drugs marijuana", "limit": 200, "print_urls": False, "no_directory": True, "chromedriver":
    "C:\\chromedriver\\chromedriver.exe", "output_directory": "drugs"}
response.download(arguments)

response = google_images_download.googleimagesdownload()
arguments = {"keywords": "upskirt", "limit": 1000, "print_urls": False, "no_directory": True, "chromedriver":
    "C:\\chromedriver\\chromedriver.exe", "output_directory": "upskirt"}
response.download(arguments)
