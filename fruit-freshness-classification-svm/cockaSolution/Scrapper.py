import os
import time
import asyncio
import logging
import aiohttp
import aiofiles
from ddgs import DDGS
from icrawler.builtin import GoogleImageCrawler, BingImageCrawler

# Reduce spam logging from icrawler
logging.getLogger().setLevel(logging.ERROR)

BASE_DIR = "/Users/Dimitrije/Documents/MachineVisionProject/Images"

# ---- QUERIES FOR EACH FRUIT / CLASS ----
QUERIES = {
    "green_apple": {
        "fresh": [
            "green apple fresh", "granny smith apple", "ripe green apple",
            "green apple close-up", "green apple hd", "healthy green apple",
        ],
        "rotten": [
            "rotten green apple", "green apple mold", "green apple fungus",
            "green apple rot", "green apple decay", "spoiled green apple",
        ]
    },

    "red_apple": {
        "fresh": [
            "fresh red apple", "red apple close-up", "ripe red apple",
            "red apple hd", "sweet red apple",
        ],
        "rotten": [
            "rotten red apple", "red apple mold", "red apple decay",
            "red apple fungus", "damaged red apple",
        ]
    },

    "orange": {
        "fresh": [
            "fresh orange fruit", "ripe orange", "orange hd",
            "orange close-up", "orange macro",
        ],
        "rotten": [
            "rotten orange", "orange mold", "orange fungus",
            "orange rot", "spoiled orange",
        ]
    },

    "banana": {
        "fresh": [
            "fresh banana", "ripe banana", "banana hd",
            "yellow banana close-up",
        ],
        "rotten": [
            "rotten banana", "banana black mold", "overripe banana",
            "banana fungus", "spoiled banana",
        ]
    },

    "kiwi": {
        "fresh": [
            "fresh kiwi", "ripe kiwi", "kiwi close-up",
            "kiwi macro", "kiwi hd",
        ],
        "rotten": [
            "rotten kiwi", "kiwi mold", "kiwi rot",
            "overripe kiwi", "spoiled kiwi",
        ]
    }
}

TARGET_PER_CLASS = 500      # what you want to aim for
CONCURRENT_DOWNLOADS = 30     # async parallel connections for ddgs downloads


# ----------------------------------------------------
# UTILITIES
# ----------------------------------------------------
def create_structure():
    """Create fruit/class folders."""
    for fruit in QUERIES:
        for cls in ["fresh", "rotten"]:
            os.makedirs(os.path.join(BASE_DIR, fruit, cls), exist_ok=True)
    print("✔ Folder structure ready.")


def count_images(path):
    """Count images by extension."""
    if not os.path.isdir(path):
        return 0
    return len([
        f for f in os.listdir(path)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])


# ----------------------------------------------------
# 1) DUCKDUCKGO (ddgs) SOURCE
# ----------------------------------------------------
def fetch_image_urls_ddg(query, max_results):
    """Fetch image URLs from DuckDuckGo via ddgs."""
    try:
        with DDGS() as ddgs:
            results_it = ddgs.images(
                query=query,
                max_results=max_results,
                safesearch="off",
            )
            results = list(results_it)
            return [item["image"] for item in results]
    except Exception as e:
        print(f"      ⚠ ddgs error for '{query}': {e}")
        return []


async def download_image(session, url, save_path):
    """Async download of a single image."""
    try:
        async with session.get(url, timeout=7) as resp:
            if resp.status != 200:
                return False
            content = await resp.read()
            async with aiofiles.open(save_path, "wb") as f:
                await f.write(content)
            return True
    except:
        return False


async def download_many_ddg(urls, save_dir, offset=0):
    """Async download many images from URL list into save_dir."""
    os.makedirs(save_dir, exist_ok=True)

    connector = aiohttp.TCPConnector(limit=CONCURRENT_DOWNLOADS)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = []
        for i, url in enumerate(urls):
            save_path = os.path.join(save_dir, f"ddg_{offset + i}.jpg")
            tasks.append(download_image(session, url, save_path))

        results = await asyncio.gather(*tasks)
        return sum(results)


# ----------------------------------------------------
# 2) GOOGLE + BING (iCrawler) SOURCES
# ----------------------------------------------------
def download_with_icrawler(queries, save_dir, needed):
    """
    Use Google + Bing via iCrawler to try to reach 'needed' images.
    Multi-threaded to speed things up.
    """
    if needed <= 0:
        return

    # we’ll be aggressive: each query tries to get a chunk
    per_query = min(max(needed * 2, 60), 150)  # try to overshoot a bit

    for q in queries:
        print(f"      🌐 iCrawler on query: {q}")

        # Google
        google = GoogleImageCrawler(
            storage={"root_dir": save_dir},
            downloader_threads=10,
            feeder_threads=1,
            parser_threads=2,
        )
        google.crawl(keyword=q, max_num=per_query)

        # Bing
        bing = BingImageCrawler(
            storage={"root_dir": save_dir},
            downloader_threads=10,
            feeder_threads=1,
            parser_threads=2,
        )
        bing.crawl(keyword=q, max_num=per_query // 2)

        current = count_images(save_dir)
        if current >= TARGET_PER_CLASS:
            print(f"      ✅ Target reached via iCrawler ({current} images).")
            break


# ----------------------------------------------------
# MAIN MULTI-SOURCE SCRAPER
# ----------------------------------------------------
async def scrape():
    create_structure()
    total_start = time.time()

    for fruit, classes in QUERIES.items():
        fruit_start = time.time()
        print(f"\n============== {fruit.upper()} ==============")

        for cls, queries in classes.items():
            cls_start = time.time()
            print(f"\n→ CLASS: {cls.upper()}")

            cls_dir = os.path.join(BASE_DIR, fruit, cls)

            # 1) DUCKDUCKGO FIRST
            all_urls = []
            print("   🔍 Collecting URLs from ddgs (DuckDuckGo)...")
            for q in queries:
                urls = fetch_image_urls_ddg(q, max_results=80)
                print(f"      {q} → {len(urls)} URLs")
                all_urls.extend(urls)

            # deduplicate URLs
            all_urls = list(set(all_urls))
            print(f"   Total unique ddgs URLs: {len(all_urls)}")

            # limit to target if too many
            urls_to_download = all_urls[:TARGET_PER_CLASS]

            before_ddg = count_images(cls_dir)
            if urls_to_download:
                print(f"   ⬇ Downloading {len(urls_to_download)} ddgs images...")
                dl_start = time.time()
                downloaded_ddg = await download_many_ddg(
                    urls_to_download, cls_dir, offset=before_ddg
                )
                dl_time = time.time() - dl_start
                print(f"   ✔ ddgs downloaded: {downloaded_ddg}/{len(urls_to_download)} "
                      f"in {dl_time:.2f} sec")
            else:
                print("   ⚠ No ddgs URLs found, skipping ddgs download.")

            # Count after ddgs
            after_ddg = count_images(cls_dir)
            print(f"   Images after ddgs: {after_ddg}")

            # 2) IF STILL NOT ENOUGH → USE ICRAWLER (GOOGLE+BING)
            if after_ddg < TARGET_PER_CLASS:
                needed = TARGET_PER_CLASS - after_ddg
                print(f"   🔁 Need {needed} more → switching to iCrawler (Google+Bing)...")
                ic_start = time.time()
                download_with_icrawler(queries, cls_dir, needed)
                ic_time = time.time() - ic_start
                final_count = count_images(cls_dir)
                print(f"   ✔ iCrawler done. Images now: {final_count} "
                      f"(iCrawler time: {ic_time:.2f} sec)")
            else:
                final_count = after_ddg

            cls_time = time.time() - cls_start
            print(f"   ⏱ CLASS {fruit}/{cls}: {cls_time:.2f} sec, "
                  f"total images: {final_count}")

        fruit_time = time.time() - fruit_start
        print(f"\n➡ TOTAL TIME FOR {fruit}: {fruit_time:.2f} sec")

    total_time = time.time() - total_start
    print("\n🔥 MULTI-SOURCE SCRAPE COMPLETE 🔥")
    print(f"⏱ TOTAL RUNTIME: {total_time:.2f} sec")
    print("Images saved in:", BASE_DIR)


if __name__ == "__main__":
    asyncio.run(scrape())
