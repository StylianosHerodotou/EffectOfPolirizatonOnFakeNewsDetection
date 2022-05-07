import os

device= "cpu"
gpus_per_trial=0;
dir_to_base="/data/pandemic_misinformation/CodeBase/EffectOfPolirizatonOnFakeNewsDetection/"
dir_to_large=os.path.join(dir_to_base, "Datasets/sag")
dir_to_ray_checkpoints=os.path.join(dir_to_base, "Ray_Tune_Checkpoints")
dir_to_ray_results=os.path.join(dir_to_base, "ray_results")



news_related_entities = [
    'CNN',    'The_New_York_Times',    'Fox_News',    'USA_Today',    'NPR',    'NBC',    'Fox_Business',    'CBS',
    'Financial_Times',    'Daily_Mail',    'The_Times',    'HBO',    'BBC',    'Fox_News_Sunday',    'PBS_NewsHour',
    'ProPublica',    'Vox_Media',    'The_New_Yorker',    'Newsmax',    'China_Daily',    'Sky_News',
    'The_Daily_Caller',    'The_Epoch_Times',    'TASS',    'The_Washington_Times',    'WNBC',    'TMZ',    'BFM_TV',
    'KCCI',    'CTV_News',    'NJ.com',    'WCVB-TV',    'Politico',    'BuzzFeed',    'The_Washington_Post',
    'News_Corp_Australia',    'The_Guardian',    'Kyodo_News',    'The_Daily_Telegraph',    'Star_Tribune',
    'The_Philadelphia_Inquirer',    'Associated_Press',    'PolitiFact',    'The_Economist',    'Houston_Chronicle',
    'Haaretz',    'The_Wall_Street_Journal',    'San_Francisco_Chronicle',    'The_Boston_Globe',    'South_China_Morning_Post',
    'Deseret_News',    'Daily_Mirror',    'NBC_News',    'CBS_News',    'The_Washington_Examiner',    'Washington_Examiner',
    'The_Texas_Tribune',    'The_Nikkei',    'The_Sun_(United_Kingdom)',    'The_Canadian_Press',    'Booth_Newspapers',
    'Sun-Sentinel',    'The_Cincinnati_Enquirer',    'El_País',    'Reuters',    'Global_Times',    'Los_Angeles_Times',
    'The_Independent',    'Chicago_Tribune',    'The_Arizona_Republic',    'People\'s_Daily',    'Asian_News_International',
    'Press_Trust_of_India',    'New_York_Daily_News',    'MailOnline',    'Florida_Today',    'The_Moscow_Times',    'CBC_News',
    'The_Seattle_Times',    'ABC_News',    'The_Hill_(newspaper)',    'American_Broadcasting_Company',    'Yahoo!_News',
    'Xinhua_News_Agency',    'Media_Matters_for_America',    'Miami_Herald',    'The_View_(talk_show)',    'Detroit_Free_Press',    'Milwaukee_Journal_Sentinel',    'Chicago_Sun-Times',    'The_Atlanta_Journal-Constitution',
    'BBC_News',    'The_Detroit_News',    'New_York_Journal-American',    'The_Dallas_Morning_News',    'The_Sunday_Times',
    'Tampa_Bay_Times',    'Wisconsin_State_Journal',    'Yonhap_News_Agency',    'The_Tennessean',    'The_San_Diego_Union-Tribune',
    'MSNBC',    'Fox_News_Channel',    'Fox_&_Friends',    'Face_the_Nation',    'New_York_Post',    'Breitbart_News',    'Bloomberg_News',    'RT_UK',    'HuffPost',    'Korean_Central_News_Agency',    'Albuquerque_Journal',    'Interfax',    'Nature_(journal)',
    'The_Des_Moines_Register',    'Variety_(magazine)',    'Orlando_Sentinel',    'Memphis_metropolitan_area',    'Idaho_Statesman',
    'New_Brunswick',    'Toronto_Star',    'Anadolu_Agency',    'The_Indian_Express',    'The_Courier-Journal',    'RBK_Group',
    'Society_of_Professional_Journalists',    'The_Straits_Times',    'The_Capital_Times',    'Libero_(newspaper)',    'Vice_Media',    'Deadline_Hollywood',
    'CNBC', 'Coronavirus_disease_2019', 'Agence_France-Presse', 'Business_Insider', 'The_Daily_Beast', 'PBS', 'RNZ_National'
]