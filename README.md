# restb-ai-fme-datathon

## Dataset

To filter the dataset, we used the following fields:

Numerical Data
- ImageData.c1c6
- ImageData.q1q6
- Characteristics.LotSizeSquareFeet
- Location.Address.PostalCodePlus4
- Property.PropertyType
- Structure.BathroomsFull
- Structure.BedroomsTotal
- Structure.BathroomsHalf
- Structure.BedroomsTotal
- Structure.Cooling
- Structure.FireplacesTotal
- Structure.LivingArea
- Structure.YearBuilt


Categorical Data
- Structure.GarageSpaces
- Structure.Heating
- Structure.NewConstructionYN


We found a lot of NAN values. To filter them, we used a pandas interpolation function. We think that this will enable us to use more data. At the end we have 99778 rows of data to use.

We choose to deal with the address using the postal code, as we achieve a higher specificity.

We shuffle the data in every epoch of the training phase to ensure robustness.

## Explainability

