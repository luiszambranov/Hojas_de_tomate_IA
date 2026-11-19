# Fotos externas (prueba de campo, Corte 3)

`plantdoc_campo/` contiene 69 fotos del split de **test** de PlantDoc
(Singh et al., 2020, *PlantDoc: A Dataset for Visual Plant Disease Detection*, CODS-COMAD;
licencia CC BY 4.0; https://github.com/pratikkayal/PlantDoc-Dataset), reorganizadas con los
nombres de clase del proyecto:

| Carpeta | Clase PlantDoc | Fotos |
|---|---|---|
| Tomato___Bacterial_spot | Tomato leaf bacterial spot | 9 |
| Tomato___Early_blight | Tomato Early blight leaf | 9 |
| Tomato___Late_blight | Tomato leaf late blight | 10 |
| Tomato___Leaf_Mold | Tomato mold leaf | 6 |
| Tomato___Septoria_leaf_spot | Tomato Septoria leaf spot | 11 |
| Tomato___healthy | Tomato leaf | 8 |
| _fuera_de_clase | Tomato leaf mosaic virus (10) + Tomato leaf yellow virus (6) | 16 |

Solo se usan para **evaluar** el salto laboratorio → campo (`src/c3_external.py`); nunca para
entrenar ni para elegir modelo. Las etiquetas son de anotadores de PlantDoc, no un diagnóstico.
