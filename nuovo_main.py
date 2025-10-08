"""
Entry point for the api
"""
from menus.main_menu import MainMenu

def main():
    """Avvia l'applicazione"""
    app = MainMenu()
    app.run()

if __name__ == "__main__":
    main()