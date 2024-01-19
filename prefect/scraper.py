import requests
from bs4 import BeautifulSoup
import time
import json 
from datetime import datetime, timedelta
import psycopg2
from prefect import task, flow

# Define 24 hours ago
twenty_four_hours_ago = datetime.now() - timedelta(days=1)



@task
def scrape(base_url): 
    """Scrape Questions from the Ask Hacker News website"""
    print("Scraping the Ask Hacker News website...")
    data = {}
    response = requests.get(base_url + "ask")
    soup = BeautifulSoup(response.text, "html.parser")
    time.sleep(5)
    question = soup.find_all(class_="titleline")
    timestamp =  soup.find_all('span', class_='age')
    for q in question:
        # for t in timestamp:
        #     t = t.get("title")
        #     delta = (datetime.now() - datetime.fromisoformat(t)) < timedelta(days=1)
        #     if delta == True: 
        title = q.text
        link = base_url + q.find("a").get("href")
        time.sleep(4.2)
        content_response = requests.get(link)
        content_soup = BeautifulSoup(content_response.text, "html.parser")
        try:
            q_body = content_soup.find(class_="toptext").text
        except: 
            q_body = None
        
        comments_list = []
        comments = content_soup.find_all(class_="comment")
        for each in comments:
            comments_list.append(each.text)
        
        data[title] = [link, q_body, comments_list]

        # data[title] = [t, link, q_body, comments_list]
    return data
    

@task
def create_schema():
    """Create the schema for the database"""
    print("Creating the schema for the database")
    connection = psycopg2.connect(host="localhost", user="root", port=5433, database="postgres", password="postgres")
    connection.autocommit = True
    cursor = connection.cursor() 

    cursor.execute("""
        BEGIN; 
        DROP TABLE IF EXISTS questions;
        CREATE TABLE IF NOT EXISTS questions (
            title TEXT,
            link TEXT,
            q_body TEXT,
            comments TEXT
        );

        COMMIT;
    """)
    cursor.close()
    connection.close()


@task
def insert_data(data):
    """Insert data into the database"""
    print("Inserting data into the database")
    connection = psycopg2.connect(host="localhost", user="root", port=5433, database="postgres", password="postgres")
    connection.autocommit = True
    cursor = connection.cursor() 

    # data 
    # with open("data.json", "r") as json_file:
    #     data = json.load(json_file)
    for key, value in data.items():
        title = key
        link = value[0]
        q_body = value[1]
        comments = json.dumps(value[2])
        cursor.execute("""
            INSERT INTO questions (title, link, q_body, comments)
            VALUES (%s, %s, %s, %s);
        """, (title, link, q_body, comments))
    cursor.close()
    connection.close()


# with Flow("HN_Scrape_Flow") as flow:
#     base_url = "https://news.ycombinator.com/"
#     data = scrape(base_url)
#     # create_schema()
#     insert_data(data, upstream_tasks=[schema_created])

@flow(flow_run_name="scraping", log_prints=True)
def prefect_flow(): 
    base_url = "https://news.ycombinator.com/"
    data = scrape(base_url)
    create_schema()
    insert_data(data)

if __name__ == "__main__":
    prefect_flow().serve(name="HN-deployment", cron="0 18 * * *")
    











