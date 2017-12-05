%Use this script to generate the probability scores for the trained model,
%currently it is set up to generate scores for the LDA classifier
T=submission_two.ClassificationDiscriminant;

%note:score is the result you want
[yfit, score]= predict(T,preprocessedtest(:,[2:133]));

submission=horzcat(preprocessedtest(:,1),score);