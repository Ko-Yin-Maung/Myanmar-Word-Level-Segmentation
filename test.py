from mig import word_segment

test = "မျိုးစေ့မမှန်လိုအပင်မသန်တာမဟုတ်မြေဆီလွှာကိုကအဆိပ်သင့်နေခဲ့တာဗျ။"
segmented_sentence = word_segment(test)
print(f"Segmented sentence: {segmented_sentence}")