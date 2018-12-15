from LanguageEngine import TextClassifier as tc
from ActionBase import FacultyActions as fa
from ActionBase import WebActions as wa
from RecommendationEngine.Code import TeacherRecommender as tr
import re

classifier = tc.Classifier()

def responceFor(input):
    predictResult = classifier.predict(input)
    questionClass = predictResult["predictions"][0]["intent"].split('.')[0]
    preprocessedQuestion = predictResult["predictions"][0]["originalSentence"]
    if(questionClass == "who"):
        return fa.action(predictResult)
    elif(questionClass == "what"):
        noOfWordsRegex = re.compile(r'In \d+ Words')
        noOfWordsRegexSearch = noOfWordsRegex.search(preprocessedQuestion)
        if noOfWordsRegexSearch is None:
            return wa.scrapeDescription(fa.extractName(preprocessedQuestion))
        else:
            return wa.scrapeDescription(fa.extractName(preprocessedQuestion), int(noOfWordsRegexSearch.group().split(' ')[1]))
    elif questionClass == "recommend":
        interests = fa.extractName(preprocessedQuestion)
        teacherName = tr.recommendTeacher(interests)
        # fa.facultyName = teacherName
        return teacherName

    return "Oops, I can't understand"
