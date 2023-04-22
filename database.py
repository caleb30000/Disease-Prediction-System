from tinydb import TinyDB, Query
User = Query()

# Database initialization
db = TinyDB('db.json')

def create_user(full_name :str, email: str, password: str):
    """
    Store complete user details in db   

    full_name:str -> Users full name   
    email:str -> Users email address   
    password:str -> Users password 

    """
    db.insert({'full_name': full_name, 'email': email, 'password': password})