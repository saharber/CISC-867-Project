#CISC 867 Project
#By Stephanie Harber
import pandas as pd
from pandas.plotting import table
import tensorflow as tf
import numpy as np
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.probability import FreqDist
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

lem = WordNetLemmatizer()


def read_file(name, file):
    with open(file, 'r') as f:
        data = f.readlines()

    stats = [
        'Season','Age','Tm','Lg','Pos','G','GS','MP','FG',
        'FGA','FG%','3P','3PA','3P%','2P','2PA','2P%','eFG%',
        'FT','FTA','FT%','ORB','DRB','TRB','AST','STL','BLK','TOV','PF','PTS']
    all_data = pd.DataFrame(columns=stats)
    data = data[1:]

    #Loopinig through each line in the file.
    for item in data:

        #Removing the new line at the end of the data.
        item = item.strip('\n')
        x = item.split(',')
                                     
        #Adding the player's information the the data frame.      
        length =len(all_data)
        all_data.loc[length]=x
    
    all_data["Name"] = name
    all_data = all_data.fillna(0)

    #Returning the input and the labels.
    return all_data


#Takes the average of all the player's statstics over their career.
def average_metrics(player):

    #Getting the average of personal fouls, points and field goal percentage. 
    pf = player['PF'].astype('float64').mean()
    points = player['PTS'].astype('float64').mean()
    fg =  player['FG%'].astype('float64').mean()

    #Calculating the pass risk using turnovers and assists.
    pass_risk =player['TOV'].astype('float').mean()/ (player['TOV'].astype('float64').mean()
                                                      +player['AST'].astype('float64').mean())
    # Returning only relevant information used for modelling.
    data =[player['Name'][0], pf,points, fg, pass_risk]
 
    return data



#Parsing Post game and between game interviews of the players.
def process_interview(file_, player, path, interviews):

    #Going through all the interviews of each player.

    total_sent =[]
    total_quest=[]
    answers =[]
    words_used =[]
    word_count =[]

    name = player.split()
    name = [i.lower() for i in name]
    lang = []
    comp_lang=[]
    

    #Getting the stop words.
    stop_words = stopwords.words("english")
    stop_words.extend(['think','going','really','thing','like','know'])
    stop_words = set(stop_words)
    
    

    count = interviews
    for i in range(1,count+1):

        fileName = path + file_ + str(i)+".txt"
        
        #Opening the current interview file.
        with open(fileName, 'r') as f:
            data = f.readlines()

        quest_count=0
        num_sentence = 0
        counter =0
        for sent in data:
            #Getting the interview sentences
            sentences=sent_tokenize(sent)
            for i in sentences:
                lang.append(i)

            if(sent[0]=="Q"):
                quest_count = quest_count+1

            else:         
                                
            
                if len(sentences)>0:
                    #Counting the number of sentences in each answer.
                    num_sentence = num_sentence+len(sentences)

                    # Looking through all of the sentences.
                    for current in sentences:

                        #Breaking down sentences into words.
                        tokenize_words = word_tokenize(current)

                        #Adding each word to the word list.
                        for word in tokenize_words:
                            word = word.lower()
                            word = lem.lemmatize(word)
                            valid = len(word)>3 and word not in name             

                            if(valid and word not in stop_words):
                                words_used.append(word)
                                counter= counter+1
                                
        #Updating the word count list when the interview has been processed.
        word_count.append(counter)                                
        total_sent.append(num_sentence)
        total_quest.append(quest_count)


    #The distribution of common words.
    fdist = FreqDist(words_used)
    #print(fdist.most_common(10))

    #Calculating the average metrics of the interviews.      
    sent_avg = sum(total_sent)/len(total_sent)
    quest_avg = sum(total_quest)/len(total_quest)
    wordCountAvg =  sum(word_count)/len(word_count)

    stats =[[player, count, round(quest_avg,2), round(sent_avg,2), round(wordCountAvg,2)]]
    col_values =["Player","Number of Interview", "Avg # Q-A pair",
                 "Avg # of sentences", "Avg # of words"]
    interview_info = pd.DataFrame(stats,columns=col_values)
           

    return interview_info, np.array(lang),data
    

def generateBinary(playerAverage, season_points):
    result =[]
    #Average number of points of a player
    averageValue =playerAverage[2]

    #Looking at the player's performance over the years.
    for i in season_points:

        i = float(i)
        
        #Points is above the career average.
        if (i >= averageValue):
            result.append(1)

        #Points is below the career average.
        else:
            result.append(0)
    
    return result
            
#Getting all the player's statstics over there careers and taking the average
dray1 = read_file("Draymond Green",'player stats/draymon_green.txt')
drayPts = dray1["PTS"]

dray  = average_metrics(dray1)

lebron1 = read_file("Lebron James",'player stats/lebron_james.txt')
lebronPts = lebron1["PTS"]
lebron = average_metrics(lebron1)

kawhi1 = read_file("Kawhi Leonard","player stats/kawhi_leonard.txt")
kawhiPts = kawhi1["PTS"]
kawhi= average_metrics(kawhi1)

kyle_low1 = read_file("Kyle Lowry", "player stats/kyle_lowry.txt")
kylePts = kyle_low1["PTS"]
kyle_low = average_metrics(kyle_low1)

russ_west1 = read_file("Russell Westbrook", 'player stats/russ_westbrook.txt')
russPts = russ_west1["PTS"]
russ_west  = average_metrics(russ_west1)

steph_curry1 = read_file("Stephen Curry",'player stats/steph_curry.txt')
stephPts = steph_curry1["PTS"]
steph_curry = average_metrics(steph_curry1)

#List of all the player's averages.
players = [dray, lebron, kawhi,  russ_west,kyle_low, steph_curry]
ptsValues = [drayPts, lebronPts, kawhiPts,  russPts,kylePts, stephPts]

count =0
binaryPoints =[] 
for i in players:
    oneHot = generateBinary(i, ptsValues[count])
    binaryPoints.append(oneHot)
print(binaryPoints)
print("Binary",len(binaryPoints))


stats = ['Name','PF','PTS','FG%','PR']
all_data = pd.DataFrame(players, columns=stats)
all_data = all_data.round(2)

#Displaying a table of the averages over their careers.
fig, ax = plt.subplots()
ax.axis('off')
ax.axis('tight')
t= ax.table(cellText=all_data.values,colWidths = [0.2]*len(all_data.columns),
            colLabels=all_data.columns,loc='center')
t.auto_set_font_size(False) 
t.set_fontsize(8)
fig.tight_layout()
plt.show()

path = "interviews/"
langVector =[]
#Reading the interviews for the players
lebron_interview,lang1, d1 = process_interview('lebron',"Lebron James",path,1)#6)

steph_interview,lang2,d2 = process_interview('steph',"Stephen Curry",path,1)#5)

dray_interview, lang3,d3 = process_interview('dray',"Draymond Green",path,1)#5)

kawhi_interview,lang4,d4 = process_interview('kawhi',"Kawhi Leonard",path,1)#)5)

kyle_interview, lang5,d5 = process_interview('kyle',"Kyle Lowry",path,1)#5)

russ_interview, lang6,d6 = process_interview('russ', "Russell Westbrook", path,1)#6)

###Plotting the break down of the interviews like number of interviews
##fig, ax = plt.subplots()
##ax.axis('off')
##ax.axis('tight')
##t= ax.table(cellText=result.values,colWidths = [0.2]*len(result.columns),
##            colLabels=result.columns,loc='center')
##t.auto_set_font_size(False) 
##t.set_fontsize(8)
##fig.tight_layout()
##plt.show()


#Getting the longest text sequence
longest = max(len(lang1),len(lang2), len(lang3),len(lang4), len(lang5), len(lang6))
print("Longest is",longest)

#Adding padding.
lang1 = np.pad(lang1,(0,abs(longest-len(lang1))))
lang2 = np.pad(lang2,(0,abs(longest-len(lang2))))
lang3 = np.pad(lang3,(0,abs(longest-len(lang3))))
lang4 = np.pad(lang4,(0,abs(longest-len(lang4))))
lang5 = np.pad(lang5,(0,abs(longest-len(lang5))))
lang6 = np.pad(lang6,(0,abs(longest-len(lang6))))

langVector = [lang1[0],lang2[0], lang3[0],lang4[0],lang5[0]]
langVector = np.array(langVector)
print("langVector",langVector.shape)


frames = [lebron_interview,steph_interview,dray_interview,
          kawhi_interview, kyle_interview, russ_interview]
result = pd.concat(frames)


vectorizer = CountVectorizer(min_df=0, lowercase=False)
data = d1+d2+d3+d4+d5+d6
vectorizer.fit(data)
vocab_size=len(vectorizer.vocabulary_)
values =vectorizer.transform(langVector).toarray()

#Neural Network
model = tf.keras.Sequential()
model.add(tf.keras.layers.Input(shape=(1271,)))
model.add(tf.keras.layers.Dense(200, activation='sigmoid'))
model.add(tf.keras.layers.Embedding(vocab_size, 4, input_length=130))
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
model.compile(optimizer='adam',
              loss='binary_crossentropy',
             metrics=['accuracy'])

model.summary()

y = all_data["PTS"][0:-1].to_numpy()
print(len(y))


model.fit(values, y,shuffle=True,
        epochs=250,
        )

test_x =vectorizer.transform([lang6[0]]).toarray()
test_y = [all_data["PTS"].iloc[-1]]
test_y = np.asarray(test_y)

score = model.evaluate (test_x, test_y)
print("Score is",score)








