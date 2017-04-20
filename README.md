# auto-keyphrase-extraction
Determine the keywords of an input document.

## About

Under normal circumstances, it is difficult to determine the keywords of a document. Some can be 1-word keywords while others may be 2-word or n-word keywords (or more appropriately, keyphrases). Furthermore, some keywords may overlap. If a document has sentences such as  "The quick brown fox jumped over the lazy dog", then plasible:
- 1-word keywords may be "fox", "dog"
- 2-word keywords (bigrams) may be "Brown Fox", "Lazy Dog" and
- 3-word keywords (trigrams) may by "quick brown fox", like so.

The tendency of a longer phrase to be a keyword is more probable. In a practical application, there is a greater probability that a search engine would fish out our fox-dog document if the search keyword was "Quick Brown Fox" rather than just "Fox" or "Dog".

The approach we use to determine the keyphrases is to get all possible candidate keyphrases and assign them as `keyphrase` or `not keyphrase`. As a result, the problem reduces to a binary classification problem that can be solved with OvO considering suitable features. The list of Features we used in our supervised analysis:

- Standard deviation
- frequency
- length
- line_position
- parabolic_position
- part_of_speech
- nth position_list

You can add/remove any feature and test the results.

## The Dataset

Directory documents contains 20 computer science articles in text format. The same documents were indexed by 15 teams of graduate and undergraduate computer science students in competitive environment.Each team's terms are stored in text format, one term per line, in files with the extension *.key, in directory teams. Note that the team numbers do not correspond to teams' performance.

When using this data set please cite the amazing contributers to the dataset:

O. Medelyan. 2009. Human-competitive automatic topic indexing. PhD thesis. Department of Computer Science, University of Waikato, New Zealand. 

O. Medelyan, I. H. Witten, D. Milne. 2008. Topic indexing with Wikipedia. In Proc. of Wikipedia and AI workshop at the AAAI-2008 Conference. Chicago, US.



## The Algorithm

1. Obtain the list of all possible keywords (monogram to 6-gram) in a `candidate_list`

2. Determine the value of the features for each candidate and store it in a dictionary. 

3. Store this dictionary in a tuple and label it as:
	- `1` if it is a keyphrase
	- `0` if it is not a keyphrase

Add this tuple to a keyword list

4. Shuffle the list of tuples 

5. Since the number of `keywords` is significantly lower than `non-keywords`, ensure we take equal number of both samples to avoid bias during the next training phase.

6. Split un-biased list into testing & training sets such that the training still remains unbiased.

7. Pass this into a classifier (SVM, Logistic Regression)

8. Determine positive/negative precision, recall & F-Measure

9. Analyse results

10. Add/remove features from step 2 and repeat the process.


## The Code

We group words into chunks called a `PHRASE` if it follows this regex pattern. 
```
(<JJ>* <NN.*>+ <IN>)? <JJ>* <NN.*>+
```
In this pattern:
- `<JJ>` is an Ajective
- `<NN.*>` reprensents different types of Nouns
- `<IN>` is a preposition

For example, "Prime Minister of the UK" would be chunked as a single `PHRASE` as it follows the regex. 
- "Prime Minsister" is a NOUN
- "of" is a preposition
- "the UK" is a NOUN

The regex for phrases in `keywords.py` can be changed to encompass better phrases (say, including Adverbs). 

Execute the following command to run the main program:

```
$ python extractor.py
```
After adding and removing certain feautres, I recorded my results in `trials.txt` for SVM. Check it out if interested. Overall, I get an 80% accuracy for determining keywords.

