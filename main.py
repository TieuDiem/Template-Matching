__all__=[]

__doc__="""

Code template matching from scratch with mode

Cosine Similarity
Correlation Cofficient

How to find the template: Find the bestest score
"""
from lib import *
from functions import create_template
from functions import brg_to_gray,template_matching
from functions import cosine_similarity,correlation_cofficient
from functions import draw_template

cwd =os.getcwd()
imgRoot =cv2.imread(os.path.join(cwd,"images","car.jpg"))
template = cv2.imread("template.jpg")
img =imgRoot.copy()


if __name__=="__main__":
    
    img_gray        = brg_to_gray(img)
    template_gray   = brg_to_gray(template)
    
    score_matrix = template_matching(img_gray,template_gray,mode="cosine_similarity")
    
    h =int(np.where(score_matrix==score_matrix.max())[0])
    w =int(np.where(score_matrix==score_matrix.max())[1])
    
    img_clone = draw_template(imgRoot,[[h,w,template_gray.shape[1],template_gray.shape[0]]])
    cv2.imshow("img",img_clone)
    cv2.waitKey(0)
    cv2.destroyAllWindows()