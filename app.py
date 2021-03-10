#packages
import streamlit as st
import streamlit.components.v1 as stc

#EDA
import pandas as pd 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity 

#Load the dataset
def load_data(data):
    df = pd.read_csv(data)
    return df

#Fxn
#Vectorize+Cosine Similarity Matrix

def vectorize_text_to_cosine_mat(data):
    count_vect = CountVectorizer(lowercase=True)
    cv_mat = count_vect.fit_transform(data)
    #get the cosine
    cosine_sim_mat = cosine_similarity(cv_mat)
    return cosine_sim_mat   

 
#Recommendation system
@st.cache    
def get_recommendation(title,cosine_sim_mat,df,num_of_rec=10):
    #indices of the course
    course_indices = pd.Series(df.index,index=df['course_title']).drop_duplicates()
    #ID of the course
    idx = course_indices[title]
    
    #look into the cosine matrix for that index
    sim_scores = list(enumerate(cosine_sim_mat[idx]))
    sim_scores = sorted(sim_scores,key=lambda x: x[1],reverse=True)
    selected_course_indices = [i[0] for i in sim_scores[1:]]    
    selected_course_scores = [i[0] for i in sim_scores[1:]]

    #Get the dataframe and title
    result_df = df.iloc[selected_course_indices]
    result_df['similarity_score'] = selected_course_scores
    final_recommended_course = result_df[['course_title','similarity_score','url','price','num_subscribers']]
    return final_recommended_course.head(num_of_rec)   


#CSS style 
RESULT_TEMP = """
<div style="width:90%;height:100%;margin:1px;padding:5px;position:relative;border-radius:5px;border-bottom-right-radius: 60px;
box-shadow:0 0 15px 5px #ccc; background-color: #f0a8a8;
  border-left: 5px solid #eef26d;">
<h4>{}</h4>
<p style="color:blue;"><span style="color:black;">📈Score::</span>{}</p>
<p style="color:blue;"><span style="color:black;">🔗</span><a href="{}",target="_blank">Link</a></p>
<p style="color:blue;"><span style="color:black;">💲Price:</span>{}</p>
<p style="color:blue;"><span style="color:black;">🧑‍🎓👨🏽‍🎓 Students:</span>{}</p>
</div>
"""
#CSS style 
RESULT_TEMP1 = """
<div style="width:90%;height:100%;margin:1px;padding:5px;position:relative;border-radius:5px;border-bottom-right-radius: 60px;
box-shadow:0 0 15px 5px #ccc; background-color: #f0a8a8;
  border-left: 5px solid #eef26d;">
<h4>{}</h4>
<p style="color:blue;"><span style="color:black;">🔗</span><a href="{}",target="_blank">Link</a></p>
<p style="color:blue;"><span style="color:black;">💲Price:</span>{}</p>
<p style="color:blue;"><span style="color:black;">🧑‍🎓👨🏽‍🎓 Students:</span>{}</p>
</div>
"""

#Search for course
@st.cache
def search_term_if_not_found(term,df,num_of_rec):
    result_df = df[df['course_title'].str.contains(term)].head(num_of_rec)
    return result_df
    
def main():

    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Udemy Course Recommender App </h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    
    menu = ["Home","Recommend","About"]
    choice = st.sidebar.selectbox("Menu",menu)
    
    df = load_data("udemy_courses.csv")
    
    if choice == "Home":
        st.subheader("Home")
        st.dataframe(df.head(10))
        
    elif choice == "Recommend":
        st.subheader("Recommend Courses")
        cosine_sim_mat = vectorize_text_to_cosine_mat(df['course_title'])
        search_term = st.text_input("Course title")
        num_of_rec = st.sidebar.number_input("Number of Recommendations",4,30,7)
        if st.button("Recommend"):
                if search_term is not None:
                        try: 
                                results = get_recommendation(search_term,cosine_sim_mat,df,num_of_rec)
                            
                                for row in results.iterrows():
                                        rec_title = row[1][0]
                                        rec_score = row[1][1]
                                        rec_url = row[1][2]
                                        rec_price = row[1][3]
                                        rec_num_sub = row[1][4]
                   
                                        stc.html(RESULT_TEMP.format(rec_title,rec_score,rec_url,rec_price,rec_num_sub),height=250)
                        except:
                                result_df = search_term_if_not_found(search_term,df,num_of_rec)
                                
                                for row in result_df.iterrows():
                                        rec_title = row[1][1]
                                        rec_url = row[1][2]
                                        rec_price = row[1][4]
                                        rec_num_sub = row[1][5]
                                
                                stc.html(RESULT_TEMP1.format(rec_title,rec_url,rec_price,rec_num_sub),height=250)

               
    else:
        st.subheader("About")
        st.text("Build with Streamlit & Pandas")


if  __name__== '__main__':
    main()
