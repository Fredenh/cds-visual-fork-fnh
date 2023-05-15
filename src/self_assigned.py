from google_images_download import google_images_download

def scrape_racket(keyword, limit):
    response = google_images_download.googleimagesdownload()
    arguments = {"keywords": keyword, "limit": limit, "print_urls": False, "size": "medium",
    "type": "photo", "chromedriver": "/Applications/Google Chrome.app."}
    try:
        paths = response.download(arguments)
        return paths[0][keyword]
    except Exception as e:
        print("Error occurred:", str(e))
        return []

keyword = "Tennis sport"
limit = 200  # Number of images to scrape
image_paths = scrape_racket(keyword, limit)


