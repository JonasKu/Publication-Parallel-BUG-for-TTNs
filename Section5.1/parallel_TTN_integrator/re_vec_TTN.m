function [B] = re_vec_TTN(A_vec,X,num)

B = X;
B{end} = A_vec{1};
A_vec = A_vec(2:end);
m = length(B) - 2;

for ii=1:m
    if iscell(B{ii}) == 1
        if iscell(num{ii}) == 1
            tmp = num{ii}{end};
        else
            tmp = num{ii};
        end
        B{ii} = re_vec_TTN(A_vec(1:tmp),X{ii},num{ii});
        A_vec = A_vec(tmp+1:end);
    else
        B{ii} = A_vec{1};
        A_vec = A_vec(2:end);
    end
    
end


% B = cell(1,1);
% count = 1;
% 
% if iscell(A_vec)==1
%     m = length(A_vec) - 2;
%     for ii=1:m
%         if iscell(A_vec{ii})
%             B{1,ii} = A_vec{ii}{end};
%             count = count + 1;
%         else
%             B{1,ii} = A_vec{ii};
%             count = count + 1;
%         end
%     end
% end
% for ii=1:m
%         if iscell(A_vec{ii}) == 1
%             tmp = vec_TTN(A_vec{ii});
%             B = [B tmp];
% %             m2 = length(A{ii}) - 2;
% %             for jj=1:m2
% %                 B_vec{ii,count} = tmp{jj};
% %                 count = count + 1;
% %             end
%         else
%             
%         end
% end   

end