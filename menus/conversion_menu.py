import questionary

class ConversionMenu:
    def __init__(self):
        print("Conversion Menu Initialized")
        pass

    def run(self):
        while True:
            choice = questionary.select(
                "Model Conversion Menu - Select an option:",
                choices=[
                    "1: Convert Model",
                    "2: Back to Main Menu"
                ]
            ).ask()

            if not choice:
                break

            choice_num = choice[0] if choice else ""

            if choice_num == "1":
                print("Converting Model...")
            elif choice_num == "2":
                break