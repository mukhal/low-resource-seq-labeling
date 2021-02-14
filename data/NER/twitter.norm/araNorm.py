# encoding: utf-8
'''--------------------------------------------------------------------------------
Script: Normalization class
Authors: Abdel-Rahim Elmadany and Muhammad Abdul-Mageed
Creation date: Novamber, 2018
Last update: Nov 22, 2019
input: text
output: normalized text
------------------------------------------------------------------------------------
Normalization functions:
- Check if text contains at least one Arabic Letter, run normalizer
- Normalize Alef and Yeh forms
- Remove Tashkeeel (diac) from Atabic text
- Reduce character repitation of > 2 characters at time
- repalce links with space
- Remove twitter username with the word USER
- replace number with NUM 
- Remove non letters or digits characters such as emoticons
------------------------------------------------------------------------------------'''
import sys
import re
# reload(sys)  # Reload does the trick!
# sys.setdefaultencoding('UTF8')


class araNorm():
    '''
            araNorm is a normalizer class for n Arabic Text
    '''

    def __init__(self):
        #print ("Welecome")
        '''
        List of normalized characters 
        '''
        self.normalize_chars = {"\u0622": "\u0627", "\u0623": "\u0627", "\u0625": "\u0627",  # All Araf forms to Alaf without hamza
                                "\u0649": "\u064A",  # ALEF MAKSURA to YAH
                                "\u0629": "\u0647"  # TEH MARBUTA to  HAH
                                }
        '''
		list of diac unicode and underscore
		'''
        self.Tashkeel_underscore_chars = {"\u0640": "_", "\u064E": 'a', "\u064F": 'u',
                                          "\u0650": 'i', "\u0651": '~', "\u0652": 'o', "\u064B": 'F', "\u064C": 'N', "\u064D": 'K'}

    def isArabicLetter(self, inputText, number_arabic_words):
        '''
        step #1: Check if text contains at least one Arabic Letter
        '''
        Arabic_chars = r'[\u0621\u0622\u0623\u0624\u0625\u0626\u0627\u0628\u0629\u062A\u062B\u062C\u062D\u062E\u062F\u0630\u0631\u0632\u0633\u0634\u0635\u0636\u0637\u0638\u0639\u063A\u0640\u0641\u0642\u0643\u0644\u0645\u0646\u0647\u0648\u0649\u064A\u064B\u064C\u064D\u064E\u064F\u0650\u0651\u0652\u0670\u0671\u067E\u0686\u06A4\u06AF]+'
        myre = re.compile(Arabic_chars)
        diacritics_list = myre.findall(inputText)
        if len(diacritics_list) >= number_arabic_words:
            # print len(diacritics_list)
            return True
        else:
            return False

    def numArabicWords(self, inputText):
        '''
        step #1: Check if text contains at least one Arabic Letter
        '''
        Arabic_chars = r'[\u0621\u0622\u0623\u0624\u0625\u0626\u0627\u0628\u0629\u062A\u062B\u062C\u062D\u062E\u062F\u0630\u0631\u0632\u0633\u0634\u0635\u0636\u0637\u0638\u0639\u063A\u0640\u0641\u0642\u0643\u0644\u0645\u0646\u0647\u0648\u0649\u064A\u064B\u064C\u064D\u064E\u064F\u0650\u0651\u0652\u0670\u0671\u067E\u0686\u06A4\u06AF]+'
        myre = re.compile(Arabic_chars)
        diacritics_list = myre.findall(inputText)
        return len(diacritics_list)

    def normalizeChar(self, inputText):
        '''
        step #2: Normalize Alef and Yeh forms
        '''
        norm = ""
        for char in inputText:
            if char in self.normalize_chars:
                norm = norm + self.normalize_chars[char]
            else:
                norm = norm + char
        return norm

    def remover_tashkeel(self, inputText):
        '''
        step #3: Remove Tashkeeel (diac) from Atabic text
        '''
        text_without_Tashkeel = ""
        for char in inputText:
            if char not in self.Tashkeel_underscore_chars:
                text_without_Tashkeel += char
        return text_without_Tashkeel

    def reduce_characters(self, inputText):
        '''
        step #4: Reduce character repitation of > 2 characters at time 
                 For example: the word 'cooooool' will convert to 'cool'
        '''
        # pattern to look for three or more repetitions of any character, including
        # newlines.
        pattern = re.compile(r"(.)\1{2,}", re.DOTALL)
        reduced_text = pattern.sub(r"\1\1", inputText)
        return reduced_text

    def replace_links(self, inputText):
        '''
        step #5: repalce links to LINK
                 For example: http://too.gl/sadsad322 will replaced to LINK
        '''
        text = re.sub('(\w+:\/\/[ ]*\S+)', '+++++++++', inputText)  # LINK
        text = re.sub('\++', 'URL', text)
        return re.sub('(URL\s*)+', ' URL ', text)

    def replace_username(self, inputText):
        '''
        step #5: Remove twitter username with the word USER
                 For example: @elmadany will replaced by space
        '''
        text = re.sub(r'(@[\u0621\u0622\u0623\u0624\u0625\u0626\u0627\u0628\u0629\u062A\u062B\u062C\u062D\u062E\u062F\u0630\u0631\u0632\u0633\u0634\u0635\u0636\u0637\u0638\u0639\u063A\u0640\u0641\u0642\u0643\u0644\u0645\u0646\u0647\u0648\u0649\u064A\u064B\u064C\u064D\u064E\u064F\u0650\u0651\u0652\u0670\u0671\u067E\u0686\u06A4\u06AF\w0-9]+)', '++++++++', inputText)  # USERNAME
        text = re.sub('\++', 'USERNAME', text)
        text = re.sub('(USERNAME\s*)+', 'USERNAME ', text)
        text = re.sub('RT USERNAME \:', 'USERNAME', text)
        return re.sub('USERNAME', ' USER ', text)

    def replace_Number(self, inputText):
        '''
        step #7: replace number with NUM 
                 For example: \d+ will replaced with NUM
        '''
        text = re.sub('[\d\.]+', 'NUM', inputText)
        return re.sub('(NUM\s*)+', ' NUM ', text)

    def replace_hashtag(self, inputText):
        '''
        step #8: replace hashtags with HASH 
                 For example: #hash will replaced with HASH
        '''
        text = re.sub(r'(#[\u0621\u0622\u0623\u0624\u0625\u0626\u0627\u0628\u0629\u062A\u062B\u062C\u062D\u062E\u062F\u0630\u0631\u0632\u0633\u0634\u0635\u0636\u0637\u0638\u0639\u063A\u0640\u0641\u0642\u0643\u0644\u0645\u0646\u0647\u0648\u0649\u064A\u064B\u064C\u064D\u064E\u064F\u0650\u0651\u0652\u0670\u0671\u067E\u0686\u06A4\u06AF_A-Za-z0-9]+)', 'HASH', inputText)  # HASH_TAGS
        return re.sub('(HASH\s*)+', ' HASH ', text)

    def remove_nonLetters_Digits(self, inputText, enabled_hashtage=False):
        '''
        step #9: Remove non letters or digits characters
                 For example: emoticons...etc
                 this step is very important for w2v  and similar models; and dictionary
        '''
        if enabled_hashtage:
            p1 = re.compile('[^\w\d#]', re.IGNORECASE | re.UNICODE)
        else:
            p1 = re.compile('[^\w\d]', re.IGNORECASE | re.UNICODE)
        sent = re.sub(p1, ' ', inputText)
        p1 = re.compile('\s+')
        sent = re.sub(p1, ' ', sent)
        return sent

    def retrive_diacritic_words(self, inputText):
        Arabic_chars = r'[\u0621\u0622\u0623\u0624\u0625\u0626\u0627\u0628\u0629\u062A\u062B\u062C\u062D\u062E\u062F\u0630\u0631\u0632\u0633\u0634\u0635\u0636\u0637\u0638\u0639\u063A\u0640\u0641\u0642\u0643\u0644\u0645\u0646\u0647\u0648\u0649\u064A\u064B\u064C\u064D\u064E\u064F\u0650\u0651\u0652\u0670\u0671\u067E\u0686\u06A4\u06AF\u0640\u064E\u064F]*'
        diac_signs = r'[\u064E\u064F\u0650\u0651\u0652\u064B\u064C\u064D]+'
        myre = re.compile(Arabic_chars+diac_signs+Arabic_chars)
        diacritics_list = myre.findall(inputText)
        return (' '.join(diacritics_list).decode('utf-8'), len(diacritics_list))

    def run(self, text, enabled_hashtage, ArabicWordsNum):
        normtext = ""
        if self.isArabicLetter(text, ArabicWordsNum):
            text = self.normalizeChar(text)
            text = self.remover_tashkeel(text)
            text = self.reduce_characters(text)
            text = self.replace_links(text)
            text = self.replace_username(text)
            text = self.replace_Number(text)
            #if not enabled_hashtage:
            #    text = self.replace_hashtag(text)
            text = self.remove_nonLetters_Digits(text, enabled_hashtage)
            text = re.sub('\s+', ' ', text.strip())
            text = re.sub('\s+$', '', text.strip())
            normtext = re.sub('^\s+', '', text.strip())
        return normtext


###############################################################
'''
Please comment below lines if used it in your package
This is ONLY an example of how to use 

if __name__ == "__main__":
	norm = araNorm()
	Fwriter=open("newInput.norm",'w')
	# We used with open to reduce memory usage (as readline function)
	with open("in.txt",'r') as Fread:
		for line in Fread:
			Fwriter.write(line+"\n")
			cleaned_line=norm.run(line.decode('utf-8'), True, 3)
			Fwriter.write(cleaned_line+"\n")
			#diac_words, count=norm.run(line.decode('utf-8'))
			#Fwriter.write(diac_words+" == "+ str(count)+"\n")
	Fwriter.close()

'''
