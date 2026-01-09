import json
import requests
import os

# Configuration
API_KEY = "YOUR_API_KEY"  # Replace with your key
SEARCH_QUERY = "Rembrandt"
BASE_URL = "https://www.rijksmuseum.nl/api/en/collection"

def download():
    import requests
import os

# The search response data you provided
search_data = {
    'orderedItems': [
        {'id': 'https://id.rijksmuseum.nl/200512613', 'type': 'HumanMadeObject'},
        # ... (rest of your list)
    ]
}

def download_collection_images():
    url = "https://data.rijksmuseum.nl/search/collection"
    headers = {"Accept": 'application/ld+json;profile="https://linked.art/ns/v1/linked-art.json"'}


    all_items = []

    while url:
        print(f"Fetching: {url}")
        response = requests.get(url, headers=headers)
        data = response.json()
        
        # Add the current page of items to our list
        # In Linked Art search, items are usually in 'items' or 'member'
        page_items = data.get('orderedItems', [])
        all_items.extend(page_items)
        
        # Check if there is a next page
        # The 'next' field is usually at the root of the JSON
        next_page_info = data.get('next')
        if next_page_info:
            url = next_page_info.get('id')
        else:
            url = None  # No more pages
    return all_items

def extract(items):
    if not os.path.exists("rijks_images"):
        os.makedirs("rijks_images")
    for item in items:
        try:
            object_url = item['id']
            object_id= object_url.split('/')[-1]
        
            # 1. Get the individual object metadata
            # We append .json or use Accept headers for Linked Art JSON
            headers = {"Accept": "application/ld+json;profile=\"linked.art\""}
            obj_resp = requests.get(object_url, headers=headers)
            obj_data = obj_resp.json()

            # 2. Extract the Image Identifier from subject_of -> digitally_carried_by
            # This follows the Linked Art structure found in your previous response
            shows= obj_data.get("shows", [])
            [digital] = [requests.get(s.get("id")).json() for s in shows]
            if  "digitally_shown_by" not in digital:
                continue    
            [ids] = [d for d in digital["digitally_shown_by"]]
            [access] = requests.get(ids["id"]).json()["access_point"]
            img = requests.get(access["id"]).content

            artists = obj_data.get("produced_by", [])
            if artists:
                artists = artists.get("referred_to_by", [])
            if artists:
                artists = [a["content"] for a in artists]

            title = obj_data.get("identified_by", [])
            if title:
                title = [t["content"] for t in title if t["type"] == "Name"]
                #assume dutch - eng sequence
                # title = title[1::2]
            [show] = shows
            artist = artists[0]
            title = title[1]
            if img:
                filename = f"rijks_images/{object_id}.jpg"
                with open(filename, "wb") as f:
                    f.write(img)
                    print(f"Downloaded: {filename}")
            else:
                print(f"No image found for {object_id}")
            #write json file with artist and title
            with open(f"rijks_images/{object_id}.json", "w") as f:
                json.dump({"artist": artist, "title": title}, f)
            print(f"Wrote metadata for {object_id}")
        except Exception as e:
            print(f"Error processing {object_id}: {e}")





def fetch_art_data(query):
    # 1. Search the collection
    params = {
        "creator": "Rembrandt",
    }
    

    items=download_collection_images()
    extract(items)




if __name__ == "__main__":
    fetch_art_data(SEARCH_QUERY)
