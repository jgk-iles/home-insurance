# Assumptions

## Preprocessing

1. If no amount (literal number) is found during parsing then NaN is returned
2. If no currency string is found during parsing then GBP is assumed to be the currency
3. Both tables are in the same order (I've checked) but the mortgage table is longer, so we just need to concat them together and chop off those extra rows we're getting from mortgages. Ideally there would be a common key on both tables but there isn't so this may have to do for now. 
4. There is a `marital_status` column in the campaign table and a `relationship` column in the other. From my inspection `maritial_status` seems more detailed - giving an indication if a customer is divorced rather than just "not in a relationship". Drop the second column. 
5. There are some unuable columns in both tables due to data protection and ethical concerns. They are listed below.
6. Can see from inspection that `education_num` and `education` from the Campaign table are basically the same information - there might be some minor differences if a person has stayed en extra year or something but for the sake of this exercise I'm going to assume the difference is negligible and drop the `education` table
7. `new_mortgage` is always yes for customers who have information on whether or not they created and account, so we need to drop this one too. 
8. `demographic_charcteristic` is usable. There are other columns like `sex` and `ethnicity` which may be innapropriate for model solection because they are too direct and may raise some questions from a regulartor or customer. When asked why a particular customer has been advertised too it would be difficult to justify saying "because you are a woman from the Carribean" but much more appropriate to say that they were selected based on "their demographic, which has shown to have a greater interest in home insurance products compared with others."


### Dropped columns

Campaign table:

* Names - privacy and not useful for model build
* Postcode - Useful but has been shown to be a proxy for race so not ethical
* Company email - Not useful for model build

Mortgage table:

* Name - privacy and not useful for model build
* Town - maybe useful but again probably a proxy for some socio-economic feature
* Paye - not useful
* Sex - not ethical to use directly in model but we may with to carry out fairness analysis
* Religion - not ethical to use directly in model but we may with to carry out fairness analysis
* Relationship - better relationship column already exists in campaign table
* Race - not ethical to use directly in model but we may with to carry out fairness analysis
* Native country - not ethical to use directly in model but we may with to carry out fairness analysis
