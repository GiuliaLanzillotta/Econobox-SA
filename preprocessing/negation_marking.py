import re

def negate2(text):
    """
    negates a sentence if contains a negative particle.

    """
    negation = False
    result = []
    words = text.split()
    for word in words:
        if any(neg == word for neg in ["not", "no", "dont",'never','wont','doesnt','didnt']):
            negation = True
    
    for word in words:
        
        if(negation == True):
            negated = "not_" + word
            result.append(negated)
            
    if(result == []):
        result = words
    return result
    
def divid(str_list, boolean = 0):
    """
    recursive function that, given a sentence, negates every subordinate clause that contains a negation. 
    EX: "yesterday I went there but I did not find you, I wonder where you were"
     ----> "yesterday I went there but not_I not_did not_not not_find not_you, I wonder where you were"
    It takes as input a string (str_list) and returns a list
    It is reccomended to carry out the function RegexpReplacer().replace on the text priorly, so to have
    a text that contains the forms 'don't' and 'can't' expressed as 'do not' and 'can not'
    
    boolean = needs to be 0 when running the function so that during the first step adjusts the spaces between commas,periods and words.

    """
    conjunctions = ['and','that','why','but','because', 'even if', "?",". .",
                    ".." 'if', 'then',"which", "who", "when", "what","where",
                    "whose", ",",";","...",".", "!", "! !", "! ! !", "if"]
    
    if(boolean == 0):
        pattern = re.compile("^\s+|\s*,\s*|\s+$")
        str_list = [x for x in pattern.split(str_list) if x]
        str_list = ' , '.join(str_list)
        pattern = re.compile("^\s+|\s*!\s*|\s+$")
        str_list = [x for x in pattern.split(str_list) if x]
        str_list = " ! ".join(str_list)
        str_list = " . ".join(re.split("\.\s+",str_list))
        str_list = str_list.replace("?", " ?")
        str_list = str_list.split()
    
     
    for i,word in enumerate(str_list):
        if(word in conjunctions):
            str_list[i+1:] = divid(str_list[i+1:],1)
            str_list[:i] = negate2(' '.join(str_list[:i]) )
            return str_list
    str_list = negate2(' '.join(str_list[:]) )
    return str_list   


def negate_text(name_input, name_output):
    """
    given a text location, it opens the texts, it negates it and saves it.  
    @ name_input = location of the input text
    @ name_output = name of the output text
    """
    text2 = []
    with open(name_input, "r") as f:
        for i,line in enumerate(f):
            text2.append(line)
    file_new = []
    for i in range(0,100000):
        file_new.append(' '.join(divid(text2[i])) + " \n")
    with open(name_output, "w") as f:
        f.writelines(file_new)

