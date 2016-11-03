#Starter code for spam filter assignment in EECS349 Machine Learning
#Author: Sonia Nigam

import sys
import numpy as np
import os
import math
import shutil
import matplotlib.pyplot as plt


def parse(text_file):
	#This function parses the text_file passed into it into a set of words. Right now it just splits up the file by blank spaces, and returns the set of unique strings used in the file. 
	content = text_file.read()
	return np.unique(content.split())

def writedictionary(dictionary, dictionary_filename):
	#Don't edit this function. It writes the dictionary to an output file.
	output = open(dictionary_filename, 'w')
	header = 'word\tP[word|spam]\tP[word|ham]\n'
	output.write(header)
	for k in dictionary:
		line = '{0}\t{1}\t{2}\n'.format(k, str(dictionary[k]['spam']), str(dictionary[k]['ham']))
		output.write(line)
		

def makedictionary(spam_directory, ham_directory, spam, ham, dictionary_filename):
	#Making the dictionary. 
	
	spam_prior_probability = len(spam)/float((len(spam) + len(ham)))
	
	words = {}

	#These for loops walk through the files and construct the dictionary. The dictionary, words, is constructed so that words[word]['spam'] gives the probability of observing that word, given we have a spam document P(word|spam), and words[word]['ham'] gives the probability of observing that word, given a hamd document P(word|ham). Right now, all it does is initialize both probabilities to 0. TODO: add code that puts in your estimates for P(word|spam) and P(word|ham).
    #iterate through each file in spam and track frequency of spam words 
	for s in spam:
		spam_tracker = []
		for word in parse(open(spam_directory + s)):
            #new entry in the dict
			if word not in words:
				words[word] = {'spam': 1, 'ham': 0}
                #increment frequency in dict
			elif word not in spam_tracker:
				spam_tracker.append(word)
				words[word]['spam'] += 1
      
    #iterate through each file in ham and track frequency of ham words          
	for h in ham:
		ham_tracker = []
		for word in parse(open(ham_directory + h)):
            #new entry in dict
			if word not in words:
				words[word] = {'spam': 0, 'ham': 1}
                #increment frequency in dict
			elif word not in ham_tracker:
				ham_tracker.append(word)
				words[word]['ham'] += 1


	for word in words:
		words[word]['spam'] = float(words[word]['spam']+1)/(float(len(spam))+1)
		words[word]['ham'] = float(words[word]['ham']+1)/(float(len(ham))+1)

    
	#Write it to a dictionary output file.
	writedictionary(words, dictionary_filename)
	
	return words, spam_prior_probability

def is_spam(content, dictionary, spam_prior_probability):
	#TODO: Update this function. Right now, all it does is checks whether the spam_prior_probability is more than half the data. If it is, it says spam for everything. Else, it says ham for everything. You need to update it to make it use the dictionary and the content of the mail. Here is where your naive Bayes classifier goes.
    
    spam_probability = math.log(spam_prior_probability)
    ham_probability = math.log(1-spam_prior_probability)

    for word in content:
        if word in dictionary: 
            spam_probability += math.log(dictionary[word]['spam'])
            ham_probability += math.log(dictionary[word]['ham'])
        else:
            pass

    if spam_probability >= ham_probability:
		return True
    else:
		return False

def spamsort(spam_directory, ham_directory, spam_mail, ham_mail, dictionary, spam_prior_probability):

	true_spam = 0
	true_ham = 0
	false_spam = 0
	false_ham = 0

	prior_true_spam = 0
	prior_true_ham = 0
	prior_false_spam = 0
	prior_false_ham = 0

	for m in spam_mail:
		content = parse(open(spam_directory + m))
		spam = is_spam(content, dictionary, spam_prior_probability)

		if spam:
			true_spam += 1
		else:
			false_ham += 1

		if spam_prior_probability >= .5:
			prior_true_spam += 1
		else:
			prior_false_ham += 1


	for m in ham_mail:
		content = parse(open(ham_directory + m))
		spam = is_spam(content, dictionary, spam_prior_probability)
		if spam:
			false_spam +=1
		else:
			true_ham += 1

		if spam_prior_probability >= .5:
			prior_false_spam += 1
		else:
			prior_true_ham += 1

	#print true_ham, true_spam, false_spam, false_ham, prior_true_ham, prior_true_spam, prior_false_ham, prior_false_spam

	spam_precision = float(true_spam)/ float(true_spam + false_spam)
	spam_recall = float(true_spam)/float(true_spam + false_ham)
	spam_f_measure = 2*((spam_precision*spam_recall)/(spam_precision+spam_recall))

	prior_spam_precision = float(prior_true_spam)/float(prior_true_spam + prior_false_spam)
	prior_spam_recall = float(prior_true_spam)/float(prior_true_spam + prior_false_ham)
	prior_spam_f_measure = 2*((prior_spam_precision*prior_spam_recall)/(prior_spam_precision+prior_spam_recall))

	ham_precision = float(true_ham)/ float(true_ham + false_ham)
	ham_recall = float(true_ham)/float(true_ham + false_spam)
	ham_f_measure = 2*((ham_precision*ham_recall)/(ham_precision+ham_recall))

	try:
		prior_ham_precision = float(prior_true_ham)/ float(prior_true_ham + prior_false_ham)
	except ZeroDivisionError:
		prior_ham_precision = 0
	try:
		prior_ham_recall = float(prior_true_ham)/float(prior_true_ham + prior_false_spam)
	except ZeroDivisionError:
		prior_ham_recall = 0
	try:
		prior_ham_f_measure = 2*((prior_ham_precision*prior_ham_recall)/(prior_ham_precision+prior_ham_recall))
	except ZeroDivisionError:
		prior_ham_f_measure = 0

	error_measure =  (spam_f_measure + ham_f_measure)/2
	prior_error_measure =  (prior_spam_f_measure + prior_ham_f_measure)/2


	return error_measure, prior_error_measure



if __name__ == "__main__":
	#Here you can test your functions. Pass it a training_spam_directory, a training_ham_directory, and a mail_directory that is filled with unsorted mail on the command line. It will create two directories in the directory where this file exists: sorted_spam, and sorted_ham. The files will show up  in this directories according to the algorithm you developed.
	spam_directory = sys.argv[1]
	ham_directory = sys.argv[2]
	error_measure = []
	prior_error_measure = []
	X = []

	
	spam = np.array([f for f in os.listdir(spam_directory) if os.path.isfile(os.path.join(spam_directory, f))])
	ham = np.array([f for f in os.listdir(ham_directory) if os.path.isfile(os.path.join(ham_directory, f))])

	spam_sets = np.split(spam, 50)
	ham_sets = np.split(ham, 50)

	
	for x in xrange(50):
		dictionary_filename = "dictionary_exp.dict"

		training_spam_directory = np.array(spam_sets[:x] + spam_sets[x+1:]).flatten()
		training_ham_directory = np.array(ham_sets[:x] + ham_sets[x+1:]).flatten()

		test_spam_mail = spam_sets[x]
		test_ham_mail = ham_sets[x]

		#create the dictionary to be used
		dictionary, spam_prior_probability = makedictionary(spam_directory, ham_directory, training_spam_directory, training_ham_directory, dictionary_filename)
		#sort the mail
		error, prior_error = spamsort(spam_directory, ham_directory, test_spam_mail, test_ham_mail, dictionary, spam_prior_probability)
		error_measure.append(error)
		prior_error_measure.append(prior_error)

	#plot f-measure by iteration number
	for i in xrange(50):
		X.append(i)

	print error_measure
	print prior_error_measure

	mean = np.mean(error_measure)
	prior_mean = np.mean(prior_error_measure)
	print "The mean f-measure is: " + str(mean)
	print "The prior mean f-measure is: " + str(prior_mean)

	plt.plot(X, error_measure, "ro")
	plt.ylabel('F-Measure')
	plt.xlabel('Sample Number')
	plt.axis([0, 50, 0, 1.1])
	plt.title('Spam Filter Results')
	plt.savefig('spamFilter.png')

	plt.clf()

	plt.plot(X, prior_error_measure, "ro")
	plt.ylabel('F-Measure')
	plt.xlabel('Sample Number')
	plt.axis([0, 50, 0, 1.1])
	plt.title('Prior Probability Results')
	plt.savefig('prior.png')










