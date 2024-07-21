import pandas as pd
from datetime import datetime
import re
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd



def prepare_data(dataset):
    #בדיקה האם זה ליסט שקיבלנו מהמשתמש או דאטהסט של הנתונים המקוריים וביצוע טרנספורמציה
    if isinstance(dataset, list):
        # Specified column names in the desired order
        columns = ["manufactor", "Year", "model", "Hand", "Gear", "capacity_Engine", "Engine_type", "Cre_date", "City",
                   "Km"]

        # Convert the list into a pandas DataFrame
        dataset = pd.DataFrame([dataset], columns=columns)

        # Correcting specific values
        replacements = {
            'manufactor': {"Lexsus": 'לקסוס'},
            'Gear': {"אוטומט": 'אוטומטית'},
            'Engine_type': {"היבריד": 'היברידי', "טורבו דיזל": 'דיזל'},
            'City': {
                "ashdod": 'אשדוד', "Tel aviv": 'תל אביב יפו', "Tzur Natan": 'צור נתן',
                "haifa": 'חיפה', "Rehovot": 'רחובות', "jeruslem": 'ירושלים',
                "Rishon LeTsiyon": 'ראשון לציון', 'פ"ת': 'פתח תקווה',
                'פתח': 'פתח תקווה', "תל": 'תל אביב יפו', "תל אביב": 'תל אביב יפו',
                "מזכרת": 'מזכרת בתיה', "ק.אתא": 'קריית אתא', "באקה": 'באקה אל גרבייה',
                "חיפ": 'חיפה'
            }
        }
        for column, replace_dict in replacements.items():
            dataset[column] = dataset[column].replace(replace_dict)
        # dataset['capacity_Engine'] = dataset['capacity_Engine'].str.replace(',', '', regex=True)
        dataset['Km'] = dataset['Km'].str.replace(',', '', regex=True)

        # Convert 'Cre_date' and 'Repub_date' to datetime
        def to_datetime(x):
            try:
                return pd.to_datetime(x)
            except ValueError:
                return None

        dataset['Cre_date'] = dataset['Cre_date'].apply(to_datetime)

        # Correct specific model values
        model_replacements = {
            "V- CLASS": 'V-Class', "E- CLASS": 'E-Class', "S- CLASS": 'S-Class',
            "Taxi": 'C-Class', "CIVIC": 'סיוויק', "JAZZ": 'ג\'אז', "ACCORD": 'אקורד',
            "CLASS": 'C-Class', "X": 'מוקה X'
        }
        dataset['model'] = dataset['model'].replace(model_replacements)

        # Function to clean and extract the model name
        def get_model(car_name):
            car_name = re.sub(r'\(.*?\)', '', car_name)
            car_name = re.sub(r'\b(החדש|חדש|חדשה|החדשה)\b', '', car_name)
            parts = car_name.split()
            if len(parts) > 1:
                if any(char.isdigit() for char in parts[1].replace(" ", "")):
                    return ' '.join(parts[1:])
                else:
                    model = ' '.join(parts[1:]).strip()
                    model = re.sub(r'\d+', '', model).strip()
                    return model if model else parts[1]
            else:
                return car_name

        dataset['model'] = dataset['model'].apply(get_model)

        # Function to remove extra spaces and characters
        def clean_car_name(car_name):
            car_name = car_name.strip()
            car_name = re.sub(r'\s?/.*', '', car_name)
            return car_name

        dataset['model'] = dataset['model'].apply(clean_car_name)

        # חישוב גיל הרכב
        current_year = datetime.now().year
        dataset['Year'] = pd.to_numeric(dataset['Year'], errors='coerce')
        dataset['Year'] = dataset['Year'].astype(int)
        dataset['Car_Age'] = current_year - dataset['Year']

        dataset['Km'] = pd.to_numeric(dataset['Km'], errors='coerce').astype('Int64')
        dataset['Km'] = dataset['Km'].where(dataset['Km'] >= 1000, dataset['Km'] * 1000)

        # Ensure correct data types
        dataset['manufactor'] = dataset['manufactor'].astype(str)
        dataset['Year'] = dataset['Year'].astype(int)
        dataset['model'] = dataset['model'].astype(str)
        dataset['Hand'] = dataset['Hand'].astype(int)
        dataset['Gear'] = dataset['Gear'].astype('category')
        dataset['capacity_Engine'] = dataset['capacity_Engine'].astype(int)
        dataset['Engine_type'] = dataset['Engine_type'].astype('category')
        dataset['City'] = dataset['City'].astype(str)

        # regions
        north = [
            'חיפה', 'כרמיאל', 'ריינה', 'נהריה', "בית ג'ן", 'טירת כרמל', 'עכו', 'קרית אתא',
            'זרזיר', 'בית שאן', 'נוף הגליל', 'באקה אל גרבייה', 'באקה א שרקיה', 'יוקנעם',
            'מגדל העמק', 'עפולה', 'ראש פינה', 'באקה אל-גרביה', 'צפת', 'טירת הכרמל',
            'בוקעתא', 'אעבלין', 'טבריה', 'עין מאהל', 'מרר', 'נצרת עילית', 'זכרון יעקב',
            'קרית ביאליק', 'קריית אתא', 'נצרת', 'עתלית', 'קרית שמונה', 'משמר השרון',
            'אלמגור', 'כאבול', 'קרית טבעון', 'כפר מנדא', 'פוריידיס', 'עספיא', 'שריד',
            'אילת השחר', "סח'נין", 'כפר תבור', 'ריחאניה', 'יובלים', 'פוריה', 'אום אל פחם',
            'עראבה', 'גילון', 'תמרת', 'שפרעם', 'ערערה', 'נשר', 'דאלית אל כרמל', 'קריית ביאליק',
            'החותרים', 'אבו סנאן', 'נחף', "מג'ד אל-כרום", 'מגאר', 'מעלות תרשיחא', 'אילון',
            'חד נס', 'אבני איתן', 'ארבל', 'רכסים', 'כפר כנא', "בית ג'אן", 'קצרין', 'נווה אור',
            'גבעת אלה', 'פקיעין', 'כמון', 'מעלות', 'נהרייה', 'דבוריה', 'כסרא', 'בית קשת',
            'טמרה', 'קרית ים', 'כפר קרע', 'מזרעה', 'מצפה נטופה ד.נ. גליל תחתון', 'סאגור',
            'יקנעם עילית', 'נאעורה', 'סלמה', 'מגדל', 'יוקנעם עילית', 'חדרה', "קרית מוצקין", "גבעת אבני", "קריית ים",
            "רמת מגשימים", "כפר מצר"
        ]

        judea_and_samaria = [
            'עלי זהב', 'חשמונאים', 'ביתר עילית', 'מתתיהו', 'אלעזר', 'כפר תפוח',
            'מעלה אדומים', 'אריאל', 'גבע בנימין', 'בת עין', 'קרית ארבע', "רבבה"
        ]

        south = [
            'באר שבע', 'עומר', 'שדרות', 'דימונה', 'קרית גת', 'שגב שלום', 'נחלה', 'נתיבות',
            'מושב מולדת', 'אילת', 'אופקים', 'תל שבע', 'שתולים', 'כסיפה', 'ברוש', 'מעגלים',
            'קציר', 'חורה', 'מיתר', 'להבים', 'עוצם', 'אשדוד', 'אשקלון', 'קרית מלאכי', "בניה", "שריגים"
        ]

        center = [
            'רעננה', 'אבן יהודה', 'רחובות', 'ראשון לציון', 'פתח תקווה', 'ירכא', 'בית דגן',
            'בת ים', 'חולון', 'גדרה', 'נס ציונה', 'גן יבנה', 'רמת גן', 'כפר סבא', 'בני ברק',
            'תל יצחק', 'תל אביב יפו', 'ראש העין', 'כפר יונה', 'גבעת שמואל', 'נתניה', 'ברקת',
            'שוהם', 'פרדס', 'גבעתיים', 'תל מונד', 'תנובות', 'אליכין', 'הוד השרון',
            'הרצליה', 'זמר', 'מכמורת', 'בית עוזיאל', 'אור עקיבא', 'אודים', 'קיסריה',
            'חרוצים', 'כפר יעבץ', 'בנימינה גבעת עדה', 'לוד', 'עזריאל',
            'מודיעין מכבים רעות', 'מודיעין', 'כפר הרי"ף', 'אלעד', 'קדרון', 'אור יהודה',
            'רמלה', 'אורנית', 'רמת השרון', 'קרית אונו', 'שער אפרים', 'זכריה',
            'פרדס חנה כרכור', 'יהוד מונוסון', 'אחיטוב', 'אזור', 'אחיעזר', 'גבעת כ"ח',
            'תל אבייב', 'יבנה', 'גבעתי', 'יהוד', 'עטרת', 'בית', 'חיננית', 'חופית', 'טייבה',
            'גני תקווה', 'יד בנימין', 'מגשימים', 'גבעתיי', 'קריות', 'ראשון', 'באר יעקב',
            'שערי תקווה', 'בארותיים', 'עמק חפר', 'פתח תיקווה', 'מזכרת בתיה', 'כפר חב"ד',
            'צפריה', 'חריש', 'רשפון', 'גן השומרון', 'מודיעין עילית', 'קרית עקרון',
            'כוכב יאיר', 'כפר מנחם', 'טירה', 'מתן', 'ניר צבי', 'קלנסווה', 'רמת ישי',
            'קדימה צורן', 'כפר עגר', 'סלעית', 'גבעת עדה', 'צור נתן', 'נגבה', 'נתנייה',
            'כפר קאסם', "צור יצחק", "טייבה משולש", "גבעת חיים מאוחד"
        ]

        jerusalem_and_surroundings = [
            'ירושלים', 'מבשרת ציון', 'בית שמש', 'צור הדסה', 'קרית יערים', 'אבו גוש',
            'עזריה', 'אורה', 'בית זית', 'פסגת זאב', 'גבעת זאב'
        ]

        def map_city_to_region(city):
            if city in north:
                return 'צפון'
            elif city in judea_and_samaria:
                return 'יהודה ושומרון'
            elif city in south:
                return 'דרום'
            elif city in center:
                return 'מרכז'
            elif city in jerusalem_and_surroundings:
                return 'ירושלים והסביבה'
            else:
                return 'לא ידוע'

        # בהנחה שה-DataFrame שלכם נקרא df ויש לו עמודה בשם 'City'
        dataset['Region'] = dataset['City'].apply(map_city_to_region)

        # חישוב יחס קילומטראז' לגיל הרכב
        dataset['Km_per_Year'] = dataset['Km'] / dataset['Car_Age']

        # חישוב כמה זמן עבר מאז תאריך יצירת המודעה
        dataset['Days_Since_Creation'] = (datetime.now() - dataset['Cre_date']).dt.days

        # דירוג לפי יוקרתיות
        car_brands = {
            "מרצדס": 1,
            "ב.מ.וו": 2,
            "אאודי": 3,
            "לקסוס": 4,
            "וולוו": 5,
            "פולקסווגן": 6,
            "מיני": 7,
            "טויוטה": 8,
            "הונדה": 9,
            "מאזדה": 10,
            "סובארו": 11,
            "פורד": 12,
            "ניסאן": 13,
            "שברולט": 14,
            "קרייזלר": 15,
            "יונדאי": 16,
            "קיה": 17,
            "סקודה": 18,
            "אופל": 19,
            "פיג'ו": 20,
            "סיטרואן": 21,
            "רנו": 22,
            "מיצובישי": 23,
            "סוזוקי": 24,
            "דייהטסו": 25
        }
        # החלפת שם החברה לדירוג החברה
        dataset['company_rank'] = dataset['manufactor'].map(car_brands).fillna(0).astype(int)
        dataset.drop(['manufactor'], axis=1, inplace=True)

        return    dataset[['model', 'Hand', 'Gear', 'capacity_Engine', 'Engine_type','Km', 'Car_Age', 'Region','Km_per_Year', 'Days_Since_Creation', 'company_rank']]

    else:

        # מצא ומחק כפיליות
        dataset = dataset.drop_duplicates()

        # Drop rows with NaN in specific columns
        dataset = dataset.dropna(subset=["Gear"])
        dataset = dataset.dropna(subset=["capacity_Engine"])
        dataset = dataset.dropna(subset=["Engine_type"])

        # Fill NaN values in 'Color' and replace specific values
        dataset['Color'] = dataset['Color'].fillna('חסר צבע')
        dataset['Color'] = dataset['Color'].replace("None", 'חסר צבע')

        # המיפוי לצבעים מאוחדים
        color_mapping = {
            'כחול כהה מטאלי': 'כחול',
            'כחול בהיר': 'כחול',
            'אפור מטאלי': 'אפור',
            'חסר צבע': 'חסר צבע',
            'שחור': 'שחור',
            'חום': 'חום',
            'כסוף': 'כסף',
            'לבן': 'לבן',
            'לבן מטאלי': 'לבן',
            'לבן פנינה': 'לבן',
            'אפור עכבר': 'אפור',
            'אפור': 'אפור',
            'כחול': 'כחול',
            'סגול': 'סגול',
            'אדום': 'אדום',
            'כסף מטלי': 'כסף',
            'כתום': 'כתום',
            'לבן שנהב': 'לבן',
            'סגול חציל': 'סגול',
            'כסוף מטאלי': 'כסף',
            'כחול בהיר מטאלי': 'כחול',
            'טורקיז': 'טורקיז',
            "בז'": 'בז',
            'בורדו': 'בורדו',
            'ירוק': 'ירוק',
            'שמפניה': 'לבן',
            'ירוק מטאלי': 'ירוק',
            'תכלת': 'תכלת',
            'חום מטאלי': 'חום',
            'אדום מטאלי': 'אדום',
            'כחול מטאלי': 'כחול',
            "בז' מטאלי": 'בז',
            'ורוד': 'ורוד',
            'ברונזה': 'ברונזה',
            'ירוק בהיר': 'ירוק',
            'זהב מטאלי': 'זהב',
            'תכלת מטאלי': 'תכלת',
            'זהב': 'זהב'
        }

        # פונקציה להחלפת הערכים על פי המיפוי
        def map_colors(dataset):
            dataset['Color'] = dataset['Color'].map(color_mapping).fillna(dataset['Color'])
            return dataset

        # להחיל את הפונקציה על הדאטה סט שלך
        dataset = map_colors(dataset)

        # Correcting specific values
        replacements = {
            'manufactor': {"Lexsus": 'לקסוס'},
            'Gear': {"אוטומט": 'אוטומטית'},
            'Engine_type': {"היבריד": 'היברידי', "טורבו דיזל": 'דיזל'},
            'Prev_ownership': {"אחר": 'לא מוגדר', "None": 'לא מוגדר'},
            'Curr_ownership': {"אחר": 'לא מוגדר', "None": 'לא מוגדר'},
            'City': {
                "ashdod": 'אשדוד', "Tel aviv": 'תל אביב יפו', "Tzur Natan": 'צור נתן',
                "haifa": 'חיפה', "Rehovot": 'רחובות', "jeruslem": 'ירושלים',
                "Rishon LeTsiyon": 'ראשון לציון', 'פ"ת': 'פתח תקווה',
                'פתח': 'פתח תקווה', "תל": 'תל אביב יפו', "תל אביב": 'תל אביב יפו',
                "מזכרת": 'מזכרת בתיה', "ק.אתא": 'קריית אתא', "באקה": 'באקה אל גרבייה',
                "חיפ": 'חיפה'
            }
        }
        for column, replace_dict in replacements.items():
            dataset[column] = dataset[column].replace(replace_dict)

        dataset = dataset.query('Gear != "לא מוגדר"')
        dataset['capacity_Engine'] = dataset['capacity_Engine'].str.replace(',', '', regex=True)

        # Fill NaN values in 'Prev_ownership' and 'Curr_ownership'
        dataset['Prev_ownership'] = dataset['Prev_ownership'].fillna('לא מוגדר')
        dataset['Curr_ownership'] = dataset['Curr_ownership'].fillna('לא מוגדר')

        # Filter out specific cities
        dataset = dataset.query(
            'City not in ["ראש", "הוד", "פתח תקווה,יהוד", "חד", "אבן", "כפר", "בת", "רא", "רמת","קרית","אומן"]')

        # Fill NaN values in 'Pic_num' with 0
        dataset['Pic_num'] = dataset['Pic_num'].fillna(0)

        # Remove rows where 'Km' is "None"
        dataset = dataset.query('Km != "None"')
        dataset['Km'] = dataset['Km'].str.replace(',', '', regex=True)

        # Convert 'Cre_date' and 'Repub_date' to datetime
        def to_datetime(x):
            try:
                return pd.to_datetime(x)
            except ValueError:
                return None

        dataset['Cre_date'] = dataset['Cre_date'].apply(to_datetime)
        dataset = dataset[dataset['Cre_date'].notnull()]

        dataset['Repub_date'] = dataset['Repub_date'].apply(to_datetime)
        dataset = dataset[dataset['Repub_date'].notnull()]

        # Correct specific model values
        model_replacements = {
            "V- CLASS": 'V-Class', "E- CLASS": 'E-Class', "S- CLASS": 'S-Class',
            "Taxi": 'C-Class', "CIVIC": 'סיוויק', "JAZZ": 'ג\'אז', "ACCORD": 'אקורד',
            "CLASS": 'C-Class', "X": 'מוקה X'
        }
        dataset['model'] = dataset['model'].replace(model_replacements)

        # Function to clean and extract the model name
        def get_model(car_name):
            car_name = re.sub(r'\(.*?\)', '', car_name)
            car_name = re.sub(r'\b(החדש|חדש|חדשה|החדשה)\b', '', car_name)
            parts = car_name.split()
            if len(parts) > 1:
                if any(char.isdigit() for char in parts[1].replace(" ", "")):
                    return ' '.join(parts[1:])
                else:
                    model = ' '.join(parts[1:]).strip()
                    model = re.sub(r'\d+', '', model).strip()
                    return model if model else parts[1]
            else:
                return car_name

        dataset['model'] = dataset['model'].apply(get_model)

        # Function to remove extra spaces and characters
        def clean_car_name(car_name):
            car_name = car_name.strip()
            car_name = re.sub(r'\s?/.*', '', car_name)
            return car_name

        dataset['model'] = dataset['model'].apply(clean_car_name)

        # הורדת שנים חריגות
        dataset = dataset[dataset['Year'] > 1988]

        # חישוב גיל הרכב
        current_year = datetime.now().year
        dataset['Car_Age'] = current_year - dataset['Year']

        # הורדת עמודות עם הרבה נאנים או עם הרבה שגיאות
        dataset.drop(["Supply_score", "Test", "Area"], axis=1, inplace=True)

        # Convert 'Km' to numeric and fill missing values
        dataset['Km'] = pd.to_numeric(dataset['Km'], errors='coerce').astype('Int64')
        dataset['Km'] = dataset['Km'].where(dataset['Km'] >= 1000, dataset['Km'] * 1000)
        dataset = dataset.drop(dataset[dataset['Km'] == 0].index)

        mean_km_by_year = dataset.groupby('Year')['Km'].median()
        dataset['Km'] = dataset.apply(lambda row: mean_km_by_year[row['Year']] if pd.isnull(row['Km']) else row['Km'],
                                      axis=1)

        # Ensure correct data types
        dataset['manufactor'] = dataset['manufactor'].astype(str)
        dataset['Year'] = dataset['Year'].astype(int)
        dataset['model'] = dataset['model'].astype(str)
        dataset['Hand'] = dataset['Hand'].astype(int)
        dataset['Gear'] = dataset['Gear'].astype('category')
        dataset['capacity_Engine'] = dataset['capacity_Engine'].astype(int)
        dataset['Engine_type'] = dataset['Engine_type'].astype('category')
        dataset['Prev_ownership'] = dataset['Prev_ownership'].astype('category')
        dataset['Curr_ownership'] = dataset['Curr_ownership'].astype('category')
        dataset['City'] = dataset['City'].astype(str)
        dataset['Price'] = dataset['Price'].astype(float)
        dataset['Pic_num'] = dataset['Pic_num'].astype(int)
        dataset['Description'] = dataset['Description'].astype(str)
        dataset['Color'] = dataset['Color'].astype(str)

        # regions
        north = [
            'חיפה', 'כרמיאל', 'ריינה', 'נהריה', "בית ג'ן", 'טירת כרמל', 'עכו', 'קרית אתא',
            'זרזיר', 'בית שאן', 'נוף הגליל', 'באקה אל גרבייה', 'באקה א שרקיה', 'יוקנעם',
            'מגדל העמק', 'עפולה', 'ראש פינה', 'באקה אל-גרביה', 'צפת', 'טירת הכרמל',
            'בוקעתא', 'אעבלין', 'טבריה', 'עין מאהל', 'מרר', 'נצרת עילית', 'זכרון יעקב',
            'קרית ביאליק', 'קריית אתא', 'נצרת', 'עתלית', 'קרית שמונה', 'משמר השרון',
            'אלמגור', 'כאבול', 'קרית טבעון', 'כפר מנדא', 'פוריידיס', 'עספיא', 'שריד',
            'אילת השחר', "סח'נין", 'כפר תבור', 'ריחאניה', 'יובלים', 'פוריה', 'אום אל פחם',
            'עראבה', 'גילון', 'תמרת', 'שפרעם', 'ערערה', 'נשר', 'דאלית אל כרמל', 'קריית ביאליק',
            'החותרים', 'אבו סנאן', 'נחף', "מג'ד אל-כרום", 'מגאר', 'מעלות תרשיחא', 'אילון',
            'חד נס', 'אבני איתן', 'ארבל', 'רכסים', 'כפר כנא', "בית ג'אן", 'קצרין', 'נווה אור',
            'גבעת אלה', 'פקיעין', 'כמון', 'מעלות', 'נהרייה', 'דבוריה', 'כסרא', 'בית קשת',
            'טמרה', 'קרית ים', 'כפר קרע', 'מזרעה', 'מצפה נטופה ד.נ. גליל תחתון', 'סאגור',
            'יקנעם עילית', 'נאעורה', 'סלמה', 'מגדל', 'יוקנעם עילית', 'חדרה', "קרית מוצקין", "גבעת אבני", "קריית ים",
            "רמת מגשימים", "כפר מצר"
        ]

        judea_and_samaria = [
            'עלי זהב', 'חשמונאים', 'ביתר עילית', 'מתתיהו', 'אלעזר', 'כפר תפוח',
            'מעלה אדומים', 'אריאל', 'גבע בנימין', 'בת עין', 'קרית ארבע', "רבבה"
        ]

        south = [
            'באר שבע', 'עומר', 'שדרות', 'דימונה', 'קרית גת', 'שגב שלום', 'נחלה', 'נתיבות',
            'מושב מולדת', 'אילת', 'אופקים', 'תל שבע', 'שתולים', 'כסיפה', 'ברוש', 'מעגלים',
            'קציר', 'חורה', 'מיתר', 'להבים', 'עוצם', 'אשדוד', 'אשקלון', 'קרית מלאכי', "בניה", "שריגים"
        ]

        center = [
            'רעננה', 'אבן יהודה', 'רחובות', 'ראשון לציון', 'פתח תקווה', 'ירכא', 'בית דגן',
            'בת ים', 'חולון', 'גדרה', 'נס ציונה', 'גן יבנה', 'רמת גן', 'כפר סבא', 'בני ברק',
            'תל יצחק', 'תל אביב יפו', 'ראש העין', 'כפר יונה', 'גבעת שמואל', 'נתניה', 'ברקת',
            'שוהם', 'פרדס', 'גבעתיים', 'תל מונד', 'תנובות', 'אליכין', 'הוד השרון',
            'הרצליה', 'זמר', 'מכמורת', 'בית עוזיאל', 'אור עקיבא', 'אודים', 'קיסריה',
            'חרוצים', 'כפר יעבץ', 'בנימינה גבעת עדה', 'לוד', 'עזריאל',
            'מודיעין מכבים רעות', 'מודיעין', 'כפר הרי"ף', 'אלעד', 'קדרון', 'אור יהודה',
            'רמלה', 'אורנית', 'רמת השרון', 'קרית אונו', 'שער אפרים', 'זכריה',
            'פרדס חנה כרכור', 'יהוד מונוסון', 'אחיטוב', 'אזור', 'אחיעזר', 'גבעת כ"ח',
            'תל אבייב', 'יבנה', 'גבעתי', 'יהוד', 'עטרת', 'בית', 'חיננית', 'חופית', 'טייבה',
            'גני תקווה', 'יד בנימין', 'מגשימים', 'גבעתיי', 'קריות', 'ראשון', 'באר יעקב',
            'שערי תקווה', 'בארותיים', 'עמק חפר', 'פתח תיקווה', 'מזכרת בתיה', 'כפר חב"ד',
            'צפריה', 'חריש', 'רשפון', 'גן השומרון', 'מודיעין עילית', 'קרית עקרון',
            'כוכב יאיר', 'כפר מנחם', 'טירה', 'מתן', 'ניר צבי', 'קלנסווה', 'רמת ישי',
            'קדימה צורן', 'כפר עגר', 'סלעית', 'גבעת עדה', 'צור נתן', 'נגבה', 'נתנייה',
            'כפר קאסם', "צור יצחק", "טייבה משולש", "גבעת חיים מאוחד"
        ]

        jerusalem_and_surroundings = [
            'ירושלים', 'מבשרת ציון', 'בית שמש', 'צור הדסה', 'קרית יערים', 'אבו גוש',
            'עזריה', 'אורה', 'בית זית', 'פסגת זאב', 'גבעת זאב'
        ]

        def map_city_to_region(city):
            if city in north:
                return 'צפון'
            elif city in judea_and_samaria:
                return 'יהודה ושומרון'
            elif city in south:
                return 'דרום'
            elif city in center:
                return 'מרכז'
            elif city in jerusalem_and_surroundings:
                return 'ירושלים והסביבה'
            else:
                return 'לא ידוע'

        # בהנחה שה-DataFrame שלכם נקרא df ויש לו עמודה בשם 'City'
        dataset['Region'] = dataset['City'].apply(map_city_to_region)

        # Drop columns with too many NaNs or irrelevant columns
        dataset.drop(["Prev_ownership", "Curr_ownership"], axis=1, inplace=True)
        # ערכים לא הגיוניים לאחר שבדקנו בגרפים
        dataset = dataset[dataset['capacity_Engine'] < 8000]
        dataset = dataset[dataset['capacity_Engine'] > 150]
        dataset = dataset[dataset['Km'] < 500000]

        # חישוב יחס קילומטראז' לגיל הרכב
        dataset['Km_per_Year'] = dataset['Km'] / dataset['Car_Age']

        # חישוב כמה זמן עבר מאז תאריך יצירת המודעה
        dataset['Days_Since_Creation'] = (datetime.now() - dataset['Cre_date']).dt.days

        # יצירת משתנה בינארי האם פורסם מחדש או לא
        dataset['Republished'] = (dataset['Cre_date'] != dataset['Repub_date']).astype(int)

        # לא רלוונטי למודל
        dataset.drop(['Cre_date', 'Repub_date', 'Year', "Description"], axis=1, inplace=True)

        # דירוג לפי יוקרתיות
        car_brands = {
            "מרצדס": 1,
            "ב.מ.וו": 2,
            "אאודי": 3,
            "לקסוס": 4,
            "וולוו": 5,
            "פולקסווגן": 6,
            "מיני": 7,
            "טויוטה": 8,
            "הונדה": 9,
            "מאזדה": 10,
            "סובארו": 11,
            "פורד": 12,
            "ניסאן": 13,
            "שברולט": 14,
            "קרייזלר": 15,
            "יונדאי": 16,
            "קיה": 17,
            "סקודה": 18,
            "אופל": 19,
            "פיג'ו": 20,
            "סיטרואן": 21,
            "רנו": 22,
            "מיצובישי": 23,
            "סוזוקי": 24,
            "דייהטסו": 25
        }
        # החלפת שם החברה לדירוג החברה
        dataset['company_rank'] = dataset['manufactor'].map(car_brands).fillna(0).astype(int)
        dataset.drop(['manufactor'], axis=1, inplace=True)

        return dataset
