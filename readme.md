# Chatbot
This is a topical chatbot that uses neural network classification to map input sentences to a topic to select a response.
The project is hosted with pythonanywhere.com @ http://aronstef.pythonanywhere.com/.

## Installation
To run the project locally make sure to install [Python 3.8](https://www.python.org/downloads/) and [Build Tools for Visual Studio](https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2019).
Clone this git repository and navigate to the folder and run: `pip install -r requirements.txt`

## Run Locally
To launch the flask webserver simply run:
`python run.py`
If you want to change the training, modify the `intents.json` and run:
`python run.py --train`

## IBM CIC Assignment Info
One day, you and your team get contacted by a customer service department in a hardware store (Bewhaos A/S). They want a chatbot for managing all customer inquries through their website.
The customer service bot should either be able to be imported into the website, or work as a seperate site which customers can access/get redirected to so forth they have any questions.
You decide how the bot should answer questions and where it should get it's data from. Frameworks are acceptable and encouraged.

### Roles
The focus of the task is on the backend however a frontend has been included based on templates and css from: https://github.com/Arraxx/new-chatbot/

## Tasks
All tasks are described as “user stories” - simply put a user wants to be able to do this, you come up with how they are going to do it. **You’re not meant to do all stories but as many as you can and in no particular order.**

### Conversation
- [x] Bewhaos wants the bot to be able to greet the customer upon starting the application and respond with simple politeness gestures as well as respond to goodbye.
- [x] A customer would like an easy to way to find out what the chatbot can do and how it can help him
- [ ] A customer enquires about available hammer products . Asking the chatbot, the bot returns a list of links to an assortment of hammers of different brands.
- [ ] A customer enquires about available screwdriver products. Asking the chatbot, the bot replies that no screwdrivers are in stock.
- [ ] Bewhaos would like to handle media with their chatbot. They want to be able to send images and videos through the chatbot.
- [ ] [Feel free to come up with other interesting or elaborate concversation scenarios]
### Configuration dashboard
- [ ] A customer service rep wants to inform the bot about new product lines, and enters a configuration dashboard, where he can publish a link for the new product line making it available for the bot to crawl.
- [ ] A customer asks a question for which the chatbot does not know the answer. The bot reports this to the configuration dashboard and creates a notification for the customer service team.
- [ ] A customer service rep logs in to the configuration dashboard to see new notifications from the chatbot. The customer service rep. finds a question about garden furniture which the chatbot has made a report on, and fixes the error by grouping the input request with other garden furniture questions.
### Nearest store lookup
- [ ] A customer wants to visit the store, but is unsure about the weather - he wants to ask the bot about the weather (BACKEND)
- [ ] A customer wants to know where the nearest store is
- [ ] A customer wants displayed by the bot where the nearest store is (FRONTEND) 


