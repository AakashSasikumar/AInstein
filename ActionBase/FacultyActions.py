import nltk
import random
import json
import difflib
from nltk.corpus import stopwords
from LanguageEngine.Summarizer import Gensim as gen
from LanguageEngine.Summarizer import NLTKFrequencySummarizer as nfs
from ActionBase import WebActions

facultyName = None
exclusionList = ["Me", "Email", "Id", "Address", "Her", "His", "Their", "He", "She", "Do", "Interests", "Are",
                 "Someone", "Who", "Teacher", "Person", "Faculty", "Knows", "Does"]
stopWords = set(stopwords.words("english"))
with open("KnowledgeEngine/Data/FacultyDetails.json", encoding="utf8") as jsonData:
    facultyDetails = json.load(jsonData)

inclusionList = []
for key in facultyDetails.keys():
    for interest in facultyDetails[key]["Interest"]:
        for word in nltk.word_tokenize(interest):
            if word not in inclusionList and word not in stopWords:
                inclusionList.append(word)

def action(data):
    questionClass = data["predictions"][0]["intent"]
    
    if questionClass == "who":
        return getDescription(data['predictions'][0]['originalSentence'].title())
    elif questionClass.split(".")[1] == "email":
        email = getEmail(data['predictions'][0]['originalSentence'].title())
        if email:
            return "There you go, the email address is " + email
        else:
            return defaultReply()
    elif questionClass.split(".")[1] == "interest":
        interests = getInterest(data['predictions'][0]['originalSentence'].title())
        if interests:
            return "Their interests include " + ", ".join(interests)
        else:
            return defaultReply()
    else:
        return defaultReply()

def defaultReply():
    return random.choice(["I'm sorry, I couldn't find that information for you.",
                                 "Well, I don't seem to be able to find that information for you now"])

def extractName(originalSentence):
    words = nltk.word_tokenize(originalSentence)
    posTags = nltk.pos_tag(words)
    name = ""
    for tag in posTags:
        if "NN" in tag[1]:
            if tag[0] not in exclusionList:
                name += tag[0] + " "
        elif tag[0] in inclusionList:
            name += tag[0] + " "
    
    return name.strip()


def getInterest(originalSentence):
    global facultyName
    name = extractName(originalSentence)
    if name == '' and facultyName:
        name = facultyName
    ID = findFacultyID(name)
    facultyName = name
    if ID == -1:
        return None
    else:
        return facultyDetails[str(ID)]["Interest"]


def getEmail(originalSentence):
    name = extractName(originalSentence)
    if name == '' and facultyName:
        name = facultyName
    ID = findFacultyID(name)

    if ID == -1:
        return None
    else:
        return facultyDetails[str(ID)]["Email"]

def getDescription(originalSentence):
    global facultyName
    name = extractName(originalSentence)
    facultyName = name
    ID = findFacultyID(name)

    if ID == -1:
        return scrapeDescription(name)
    
    summary = gen.getSummary(facultyDetails[str(ID)]['description'])
    if len(summary) == 0:
        return facultyDetails[str(ID)]['description']
    else:
        return summary

def findFacultyID(name):
    with open("KnowledgeEngine/Data/FacultyNames.json", encoding="utf8") as jsonData:
        teacherNamesAndID = json.load(jsonData)
    teacherNames = teacherNamesAndID.values()
    
    try:
        teacherID = list(teacherNames).index(difflib.get_close_matches(name, teacherNames, cutoff=0.5)[0]) + 1
        return teacherID
    except Exception:
        return -1

def scrapeDescription(name):
    return WebActions.scrapeDescription(name)