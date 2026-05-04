import os
import random
from datetime import datetime, timedelta

# Configuration
START_DATE = datetime(2026, 3, 19, 10, 0, 0)
END_DATE = datetime(2026, 5, 4, 18, 0, 0)
NUM_COMMITS_PER_USER = 10

USER_HARSH = {
    "name": "Harshkesharwani789",
    "email": "Atul123564@gmail.com"
}
USER_SIDDHARTHA = {
    "name": "SiddharthaShukla8",
    "email": "siddharthashuklajee8@gmail.com"
}

def generate_random_date(start, end):
    delta = end - start
    int_delta = (delta.days * 24 * 60 * 60) + delta.seconds
    random_second = random.randrange(int_delta)
    return start + timedelta(seconds=random_second)

def main():
    commits = []
    
    # Generate 10 commits for Harsh
    for i in range(NUM_COMMITS_PER_USER):
        date = generate_random_date(START_DATE, END_DATE)
        commits.append({
            "user": USER_HARSH,
            "date": date,
            "msg": f"Refactor: component enhancement module {i+1}"
        })
        
    # Generate 10 commits for Siddhartha
    for i in range(NUM_COMMITS_PER_USER):
        date = generate_random_date(START_DATE, END_DATE)
        commits.append({
            "user": USER_SIDDHARTHA,
            "date": date,
            "msg": f"Feature: analytics integration part {i+1}"
        })

    # Sort commits by date
    commits.sort(key=lambda x: x["date"])

    # Create a dummy file to modify
    dummy_file = "history_log.txt"
    
    for i, commit in enumerate(commits):
        date_str = commit["date"].strftime("%Y-%m-%dT%H:%M:%S")
        user = commit["user"]
        
        # Append to dummy file
        with open(dummy_file, "a") as f:
            f.write(f"Commit {i+1} by {user['name']} on {date_str}\n")
            
        # Git command - EXPLICITLY SETTING GIT_COMMITTER_DATE
        cmd = (
            f'export GIT_AUTHOR_NAME="{user["name"]}" && '
            f'export GIT_AUTHOR_EMAIL="{user["email"]}" && '
            f'export GIT_COMMITTER_NAME="{user["name"]}" && '
            f'export GIT_COMMITTER_EMAIL="{user["email"]}" && '
            f'export GIT_COMMITTER_DATE="{date_str}" && '
            f'git add -f {dummy_file} && '
            f'git commit --date="{date_str}" -m "{commit["msg"]}"'
        )
        
        print(f"Executing commit {i+1}/{len(commits)} for {user['name']} at {date_str}")
        os.system(cmd)

if __name__ == "__main__":
    main()
