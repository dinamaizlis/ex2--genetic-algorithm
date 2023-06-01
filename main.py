import sys, os
import random

MAX_GENERATIONS= 200
NUM_PARENTS= 100
RangeCros= NUM_PARENTS//2
ResultOk= False
LOCAL_MAXIMUM= 15
FitnessCalls= 0
alphabet= 'abcdefghijklmnopqrstuvwxyz'


with open('enc.txt', 'r') as file:
    EncryptedText= file.read()
    LenText=len(EncryptedText)

# Read dictionary of words
with open('dict.txt', 'r') as file:
    dictionary= set(file.read().lower().split())

# Read letter frequency statistics
with open('Letter_Freq.txt', 'r') as file:
    FreqFetter= file.read().lower().splitlines()
    # create dic of letter and frequency
    FreqFetterDic= {}
    for line in FreqFetter:
        freq, letter= line.split("\t")
        if freq != '':
            FreqFetterDic[letter]= float(freq)

# Read letter pair frequency statistics
with open('Letter2_Freq.txt', 'r') as file:
    FreqBigram= file.read().lower().splitlines()
    # create dic of 2letter and frequency
    FreqBigramDic= {}
    for line in FreqBigram:
        freq, letter= line.split("\t")
        if freq != '':
            FreqBigramDic[letter]= float(freq)

#creat NUM_PARENTS permotion of all the english letters
def GeneratePopulation():
    population= []
    for i in range(NUM_PARENTS):
        permutation= list(alphabet)
        random.shuffle(permutation)
        DecryptionKey= {}
        for i, letter in enumerate(alphabet):
            DecryptionKey[letter]= permutation[i]
        population.append(DecryptionKey)
    return population

# get decrypt of the text by DecryptionKey {a:s,b:g.... }
def DecryptText(EncryptedText , DecryptionKey) -> dictionary:
    DecryptedText= ""
    for letter in EncryptedText:
        if letter in DecryptionKey:
            DecryptedText += DecryptionKey[letter]
        else:
            DecryptedText += letter
    return DecryptedText

# get dic of letter : frequency -> for letter and bigram 's': 0.010301353013530135, 'w': 0.03920664206642067, 'k': 0.10716482164821649, 'c': 0.05796432964329643.....
def GotFrequencyOfLetterOrBigram(data):
    FreqLetterDic, FreqBigramDic= {} ,{}
    for letter in data:
        if letter in FreqLetterDic :
            FreqLetterDic[letter]+= 1
        else:
            FreqLetterDic[letter]= 1

    for i in range(LenText-1):
        Bigram= data[i:i+2]
        if Bigram in FreqBigramDic:
            FreqBigramDic[Bigram] += 1
        else:
            FreqBigramDic[Bigram]= 1

    FreqLetterDic= {letter: freq / LenText for letter, freq in FreqLetterDic.items()}
    FreqBigramDic= {letter: freq / LenText for letter, freq in FreqBigramDic.items()}
    return FreqLetterDic, FreqBigramDic

# select new perents -  new decrypted
def ChossePerant(population, FitnessScores):
    TournamentCandidates= random.sample(range(len(population)), 5)
    TournamentScores= [FitnessScores[i] for i in TournamentCandidates]
    return population[TournamentCandidates[TournamentScores.index(max(TournamentScores))]]

#Calculate Fitness- give a score for the decryption -
# calculate it whit the files: dic.txt , letterfreq.txt and letter2freq.txt
def CalculateFitness(DecryptionKey, EncryptedText):
    global ResultOk
    global FitnessCalls

    ResultOk= False
    FitnessCalls += 1

    DecryptedText= DecryptText(EncryptedText, DecryptionKey).replace(".", "").replace(",", "").replace(";", "")
    FreqFetter ,FreqBigram= GotFrequencyOfLetterOrBigram(DecryptedText)
    MatchingWords ,MatchingLetters= 0 ,0
    DecryptedTextSet= set(DecryptedText.split(" "))

    MatchingWords += sum([1 for word in dictionary if word in DecryptedTextSet])
    MatchingLetters +=  sum([(1 - abs(FreqFetter[letter] - FreqFetterDic[letter])) ** 2 for letter in FreqFetterDic if letter in FreqFetter and letter in FreqFetterDic])
    for Bigram in FreqBigramDic:
        if Bigram in FreqBigram and Bigram in FreqBigramDic:
            MatchingLetters += (1 - abs(FreqBigram[Bigram] - FreqBigramDic[Bigram])) ** 2.5

    if MatchingWords >= 0.98 * len(DecryptedTextSet):
        ResultOk= True

    return MatchingWords + MatchingLetters

#create new decryption by crossover of two parents
def crossover(parent1, parent2):
    keys= list(parent1.keys())
    index= random.randint(0, 25) # 26 letter in english- start from index 0

    child= {key: parent1[key] for key in keys[:index+1]} #copy from perent 1
    for i in range(index + 1, 26): #fill the othwer with the letter in perant 2
        letter= parent2[keys[i]]

        while letter in child.values():
            #Change duplicate letter with letter that dosnt apper
            child[next(k for k, v in child.items() if v== letter)]= next(l for l in parent2.values() if l not in child.values())
            letter= parent2[keys[i]]
        child[keys[i]]= letter
    return child


#handeling with the local max
def FixLocalMax(EncryptedText,BestDecryption, BestDecryptionFitness):
    for i in range(5):
        BestDecryptionkey, _, BestFitness= GeneticAlgorithm(EncryptedText)
        if BestDecryptionFitness < BestFitness:
            BestDecryptionFitness= BestFitness
            BestDecryption= BestDecryptionkey
        return BestDecryption if ResultOk else None

    return BestDecryption

#do n swap on the decryption
def LocalOptimization(individual, numberOfSwap):
    individualCopy= individual.copy()
    for i in range(numberOfSwap):
        individualCopy= mutate(individualCopy)
    return individualCopy

# Darwin Mutation
def DarwinMutation(population, FitnessScores, EncryptedText):
    for i in range(len(population)):
        if i<40:
            MIndividual= LocalOptimization(population[i],3)
        else:
            MIndividual= LocalOptimization(population[i],4)
        MFitness= CalculateFitness(MIndividual, EncryptedText)
        if MFitness > FitnessScores[i]:
            FitnessScores[i]= MFitness

# Lamarck Mutation
def LamarckMutation(population, FitnessScores, EncryptedText):
    for i in range(len(population)):
        if i<30:
            MIndividual= LocalOptimization(population[i],3)
        else:
            MIndividual= LocalOptimization(population[i],2)
        MFitness= CalculateFitness(MIndividual, EncryptedText)
        if MFitness > FitnessScores[i]:
            FitnessScores[i]= MFitness
            population[i]= MIndividual

#return mutate decryption -  swap 2 letters
def mutate(DecryptionKey):
    keys= list(DecryptionKey.keys())
    index1, index2= random.sample(range(26), 2) #26 letters in english
    DecryptionKey[keys[index1]], DecryptionKey[keys[index2]]=DecryptionKey[keys[index2]], DecryptionKey[keys[index1]]
    return DecryptionKey


#the main algoritem for getting decryption on data that we got on file enc.txt
def GeneticAlgorithm(EncryptedText, mode=''):
    population= GeneratePopulation()# 100 random permotion - population
    NotImprovement, BestFitness= 0 , 0

    for generation in range(MAX_GENERATIONS):
        FitnessScores , AllChild= [] ,[]


        for DecryptionKey in population:
            fitness= CalculateFitness(DecryptionKey, EncryptedText)
            FitnessScores.append(fitness)
            if ResultOk:
                return DecryptionKey, 0, fitness

        for i in range(RangeCros):
            parent1,parent2=ChossePerant(population, FitnessScores), ChossePerant(population, FitnessScores)
            child= mutate(crossover(parent1, parent2))
            AllChild.append(child)

        GoodPIndices= sorted(range(len(FitnessScores)), key=lambda j: FitnessScores[j], reverse=True)[:15]
        GoodACIndices= sorted(range(len(AllChild)), key=lambda j: CalculateFitness(AllChild[j], EncryptedText),reverse=True)[:15]
        BadIndices= sorted(range(len(FitnessScores)), key=lambda j: FitnessScores[j])[:3*15]

        for i in range(15):
            population[BadIndices[i]]= population[GoodPIndices[0]]
            population[BadIndices[i + 15]]= population[GoodPIndices[i]]
            population[BadIndices[i + 2 * 15]]= AllChild[GoodACIndices[i]]

        DarwinMutation(population, FitnessScores, EncryptedText) if mode== 'darwin' else None
        LamarckMutation(population, FitnessScores, EncryptedText) if mode== 'lamarck' else None

        BestDecryptionkey= population[FitnessScores.index(max(FitnessScores))]
        PriveuseBestFitness= BestFitness
        BestFitness= FitnessScores[FitnessScores.index(max(FitnessScores))]

        NotImprovement= NotImprovement + 1 if BestFitness - PriveuseBestFitness < 0.0001 else 0

        if NotImprovement== LOCAL_MAXIMUM:
            break
        print(f"Generation: {generation+1} | | Best Fitness: {BestFitness}")
    return BestDecryptionkey, NotImprovement, BestFitness


if __name__== '__main__':
    mode= ''
    args= sys.argv[1:]
    if len(args) > 0:
        encFile= args[0]
        mode= ''
        if len(args) > 1:
            mode= args[1].lower()

    BestDecryption, counter, fitness= GeneticAlgorithm(EncryptedText, mode)

    if counter== LOCAL_MAXIMUM and not ResultOk:
        BestDecryption= FixLocalMax(EncryptedText, BestDecryption, fitness)

    DecryptedText= DecryptText(EncryptedText, BestDecryption)

    # creating files
    with open("plain.txt", "w") as file:
        file.write(DecryptedText)

    with open("perm.txt", "w") as file:
        for letter, decryptedLetter in BestDecryption.items():
            file.write(f"{letter} {decryptedLetter}\n")

    print("fitness calls: ", FitnessCalls)
