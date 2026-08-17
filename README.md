# Predicting Customer Purchase Behavior with Luma Event Data

**Author:** Ishaan Singh

## Executive Summary

In this capstone, I investigated whether early customer behavior can help a digital business identify people who are likely to purchase later. I used the Luma event dataset, which captures website activity, mobile activity, email tracking, product interactions, cart actions, checkout activity, and purchase events.

I built a customer level machine learning solution that combines purchase prediction, customer segmentation, and next best action recommendations. I found that meaningful early behavioral signals are available. On a held out test set, my tuned random forest model achieved a ROC AUC of 0.819. In plain language, the model was generally good at ranking later purchasers above non purchasers using only the first five observed customer interactions.

My strongest recommendation is to use this project as a decision support prototype. Customers showing high early intent should receive timely product recommendations or cart reminders, while customers with lower intent should receive lower frequency educational content. I would validate these actions through controlled experiments before a full rollout.

## Problem Statement

Digital businesses interact with customers through websites, mobile applications, and email campaigns. The challenge is to decide which customer should receive a reminder, a product recommendation, educational content, or no message at all. Broad campaigns can waste marketing spend and send irrelevant messages.

My goal was to build a data driven solution that identifies early purchase intent, groups customers with similar behavior, and suggests an appropriate engagement action. The potential benefits are more relevant messages, better conversion outcomes, lower unnecessary marketing cost, and a better customer experience.

I addressed the following question:

**Can I use machine learning to predict customer purchase behavior and recommend personalized engagement actions that may improve conversion outcomes?**

## Model Outcomes and Predictions

I used two types of learning. For supervised learning, I built a binary classification model. Its expected output is a purchase probability and a purchase or no purchase prediction for each customer. The target is whether a customer purchased after the first five observed interactions.

For unsupervised learning, I used K means clustering to identify groups of customers with similar early behavior. I then combined the predicted probability, observed behavior, and customer segment into a simple next best action recommendation. The recommendation is an operational aid, not an automated decision.

## Data

I used the Luma event dataset supplied for this project. The source data includes 46,110 event records and 95 fields. It contains customer identifiers, timestamps, page activity, product interactions, cart actions, checkout events, purchase records, email tracking codes, device context, and geographic context.

I included the source data as `luma_post_extended.zip` so the repository stays manageable. The final notebook reads the ZIP file directly. After removing exact duplicate records and records without a usable customer identifier, I retained 23,054 usable event records representing 929 customers.

## Data Acquisition and Potential

I used the supplied Luma event dataset as the source for this project. It is a single source rather than a combination of external sources, so I do not claim to measure every driver of customer behavior. It is still suitable for this initial solution because it brings together web activity, product engagement, purchases, email tracking, application activity, and customer context at the event level.

I assessed the data's potential in the notebook with a missingness chart, purchase outcome chart, email conversion chart, behavioral distribution plots, referrer analysis, and a customer segment visualization. These views showed clear differences in early behavior between customers who later purchased and those who did not. Future versions should supplement this dataset with campaign cost, consent, customer lifetime value, and communication response data.

## My Approach

I organized the project into four connected steps.

1. **Clean and prepare the data.** I removed duplicate events, excluded events without customer IDs, converted timestamps and event counters, and created clear behavioral flags such as email touch, app activity, product interaction, cart page activity, and checkout page activity.

2. **Create a realistic prediction window.** I used the first five customer interactions as the observation period and predicted whether the customer purchased after that point. I excluded customers with fewer than five observed interactions and the five customers who purchased during the observation period. This reduced the risk that the model simply learned from completed purchase journeys.

3. **Explore and segment customers.** I used visual analysis, nonparametric statistical tests, and K means clustering to understand how early behavior differs across customers.

4. **Compare and tune classification models.** I evaluated a no skill dummy baseline, logistic regression, and random forest using stratified five fold cross validation. I then tuned the random forest with grid search and evaluated it one final time on a held out test set.

## Data Preprocessing and Preparation

I removed exact duplicate rows and records without a usable customer ID. I converted timestamps to a consistent datetime format and converted sparse event counters to numeric fields. For these counter fields, I treated missing values as zero because a blank value indicates that the event was not recorded for that row. I created features for email touch, application activity, product interaction, cart page activity, checkout page activity, unique pages, unique site sections, and early referrer type.

I aggregated the event records to one customer level record using only the first five interactions. I used a stratified 75% training and 25% test split so that the later purchase rate remained similar in both groups. Within each cross validation fold, I imputed numeric values with the median, imputed categorical values with the most frequent value, standardized numeric features, and one hot encoded categorical features. These preparation steps were placed inside the model pipeline so that the validation and test data did not influence the training transformations.

## Key Findings

My final modeling cohort contained 620 customers. Of these, 219 customers purchased after the first five interactions, for a later purchase rate of 35.3%.

Early email exposure was the clearest signal in this dataset. Customers with an email touch in the first five events had a 77.0% later purchase rate, compared with 18.6% for customers without an email touch. Product interaction, cart additions, and browsing across more pages were also associated with later purchase. The relationship between email touch and later purchase was statistically significant in a chi square test with p less than 0.001. I interpret this as a strong association, not proof that email alone caused customers to buy.

The three customer segments I selected were practical for marketing decisions:

| Segment | Customers | Later Purchase Rate | Interpretation |
|---|---:|---:|---|
| High intent shoppers | 125 | 68.8% | Strong product, cart, and email engagement |
| Web explorers | 256 | 34.8% | Moderate browsing and product exploration |
| App focused browsers | 239 | 18.4% | High app activity with limited commerce engagement |

## Modeling Strategy

I selected three classification models for different reasons. The dummy classifier establishes the performance of a no skill approach. Logistic regression provides a transparent and interpretable baseline. Random forest can learn nonlinear relationships among early engagement signals without requiring me to predefine those relationships.

I selected K means for segmentation because it creates understandable groups from standardized behavioral measures. I reviewed two through five possible cluster counts and selected three segments because the result created a practical set of engagement strategies. The notebook reports silhouette scores for each candidate count.

## Model Evaluation and Results

I used ROC AUC as the primary model metric because it measures how well a model ranks potential purchasers above non purchasers and is less misleading than accuracy when the target groups are not perfectly balanced. I also reported average precision and F1 to show the quality of positive predictions.

| Model | Mean Cross Validated ROC AUC | Mean Average Precision | Mean F1 |
|---|---:|---:|---:|
| Dummy baseline | 0.500 | 0.353 | 0.000 |
| Logistic regression | 0.833 | 0.745 | 0.691 |
| Random forest | 0.841 | 0.755 | 0.703 |

I tuned the random forest by testing its tree count, tree depth, and minimum leaf size with grid search. The best configuration used 500 trees, a maximum depth of 8, and a minimum leaf size of 1. Its validation ROC AUC was 0.839 during tuning.

On the held out test set, the tuned model achieved:

| Metric | Result | What it means |
|---|---:|---|
| ROC AUC | 0.819 | The model ranked later purchasers meaningfully above non purchasers. |
| Average precision | 0.714 | Positive predictions were substantially better than the base purchase rate. |
| Precision | 0.689 | About 69% of customers predicted to purchase did purchase later. |
| Recall | 0.764 | The model identified about 76% of later purchasers. |
| F1 | 0.724 | Precision and recall were reasonably balanced. |

Permutation importance showed that early email touches were the most useful feature for the final model. Early browsing breadth, measured by unique pages and unique site sections, was also informative. This is interpretable from a business perspective: customers who engage through more than one early touchpoint often display stronger purchase intent. It does not prove that sending more email messages will produce the same result because the observed email activity may reflect existing targeting decisions.

## Suggested Next Best Actions

I mapped model probability and early behavior to a simple engagement prototype.

| Customer situation | Recommended action |
|---|---|
| High predicted probability and cart activity | Send a cart reminder featuring saved items. |
| High predicted probability without cart activity | Send a personalized product recommendation. |
| Medium predicted probability without an email touch | Offer opt in email content or useful product education. |
| Medium predicted probability with an email touch | Send comparison content or relevant product education. |
| Low predicted probability | Use low frequency awareness content and avoid unnecessary discounting. |

This decisioning logic is intentionally simple and transparent. It gives a marketing team a starting point for targeting rules while preserving room for business judgment and customer communication preferences.

## Limitations

This project uses observational event data, so I cannot claim that any feature or marketing touchpoint caused a purchase. The email relationship may partly reflect prior targeting of customers who already had stronger intent. The data is also a modest sample and does not include a complete view of customer value, costs, or communication consent.

I reduced target leakage by limiting features to the first five events, but a production version should define an explicit prediction horizon, such as purchase within seven days after a scored session. The next best action recommendations have not been tested with live customers and should not be automated without an experiment.

## Recommendations and Next Steps

1. I recommend testing cart reminders and personalized product recommendations with a randomized A/B experiment for high probability customers.

2. I recommend measuring incremental conversion, revenue, unsubscribe rate, and customer satisfaction rather than conversion alone.

3. I would create a time based training and testing design that predicts purchase within a clearly defined future window.

4. I would add customer lifetime value and campaign cost data so the system can optimize profitable engagement rather than purchase occurrence only.

5. I would calibrate predicted probabilities and select thresholds based on the cost of discounts, messages, and missed opportunities.

## Repository Contents

| File | Purpose |
|---|---|
| [luma_customer_purchase_capstone.ipynb](luma_customer_purchase_capstone.ipynb) | Fully executed technical notebook with cleaning, EDA, statistical testing, clustering, cross validation, grid search, model interpretation, and next best action logic. |
| [luma_post_extended.zip](luma_post_extended.zip) | Compressed project dataset read directly by the notebook. |
| [README.md](README.md) | This nontechnical capstone report. |

## How to Run the Notebook

I developed the notebook with Python 3.12. It uses `pandas`, `numpy`, `matplotlib`, `seaborn`, `scipy`, and `scikit-learn`. Keep the ZIP file in the repository root, open the notebook, and run all cells from top to bottom.
