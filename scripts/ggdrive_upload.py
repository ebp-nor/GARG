"""Upload a file to Google Drive using the Google Drive API.

Create a new project in the Google Cloud Console and enable the Google
Drive API.

1. go to https://console.developers.google.com/
2. Create a new project
3. Enable the Google Drive API (APIs & Services -> Library)
4. Create credentials for the project (APIs & Services -> Credentials
   -> Create Credentials)
5. Download the credentials as a JSON file

"""

import argparse
import os
import pickle

from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

# If modifying these SCOPES, delete the file token.pickle.
SCOPES = ["https://www.googleapis.com/auth/drive.file"]

# If uploading files to a folder set GOOGLE_DRIVE_PARENT_ID to the
# folder FILE_ID
GOOGLE_DRIVE_PARENT_ID = os.environ.get("GOOGLE_DRIVE_PARENT_ID")


def authenticate(auth_path):
    """Authenticate the user and return the Google Drive service."""
    creds = None
    # Token file for storing user access and refresh tokens.
    token_path = "token.pickle"

    # Load existing tokens
    if os.path.exists(token_path):
        with open(token_path, "rb") as token:
            creds = pickle.load(token)

    # If there are no valid credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(auth_path, SCOPES)
            creds = flow.run_local_server(port=0)

        # Save the credentials for the next run
        with open(token_path, "wb") as token:
            pickle.dump(creds, token)

    service = build("drive", "v3", credentials=creds)
    return service


def upload_or_update_file(service, file_path, parent_id=None):
    """Upload or update a file to Google Drive."""
    file_name = os.path.basename(file_path)
    existing_file = find_file(service, file_name, parent_id)

    file_metadata = {"name": os.path.basename(file_path)}

    media = MediaFileUpload(file_path, resumable=True)

    if existing_file:
        print(f'Updating file with ID: {existing_file["id"]}')
        file_id = existing_file["id"]

        # Update the file content
        file = (
            service.files()
            .update(fileId=file_id, media_body=media, fields="id")
            .execute()
        )

        current_parents = existing_file.get("parents", [])
        # Remove the file from all current parents, then move to the new parent if provided
        if parent_id and parent_id not in current_parents:
            remove_parents = ",".join(current_parents)
            file = (
                service.files()
                .update(
                    fileId=file_id,
                    addParents=parent_id,
                    removeParents=remove_parents,
                    fields="id, parents",
                )
                .execute()
            )
    else:
        print("Uploading new file")
        if parent_id:
            file_metadata["parents"] = [parent_id]
        file = (
            service.files()
            .create(body=file_metadata, media_body=media, fields="id")
            .execute()
        )

    print(f'File ID: {file.get("id")}')


def find_file(service, file_name, parent_id=None):
    """Find a file in Google Drive by name and parent folder ID."""
    query = f"name='{file_name}' and trashed=false"
    if parent_id:
        query += f" and '{parent_id}' in parents"

    results = (
        service.files()
        .list(q=query, spaces="drive", fields="files(id, name)")
        .execute()
    )

    items = results.get("files", [])
    return items[0] if items else None


def main():
    parser = argparse.ArgumentParser(description="Upload a file to Google Drive")
    parser.add_argument("file_path", help="Path to the file to upload")
    parser.add_argument("auth_file", help="Path to the authentication JSON file")
    parser.add_argument(
        "--parent_id",
        help="ID of the parent directory/folder in Google Drive",
        default=GOOGLE_DRIVE_PARENT_ID,
    )

    args = parser.parse_args()

    service = authenticate(args.auth_file)
    upload_or_update_file(service, args.file_path, args.parent_id)


if __name__ == "__main__":
    main()
