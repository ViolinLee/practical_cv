function main()
boxes=[200,200,400,400,0.99;
        220,220,420,420,0.9;
        100,100,150,150,0.82;
        200,240,400,440,0.5];
overlap=0.8;
pick = NMS(boxes, overlap);
figure;
for (i=1:size(boxes,1))
    rectangle('Position',[boxes(i,1),boxes(i,2),boxes(i,3)-boxes(i,1),boxes(i,4)-boxes(i,2)],'EdgeColor','y','LineWidth',6);
    text(boxes(i,1),boxes(i,2),num2str(boxes(i,5)),'FontSize',14,'color','b');
end
for (i=1:size(pick,1))
    rectangle('Position',[boxes(pick(i),1),boxes(pick(i),2),boxes(pick(i),3)-boxes(pick(i),1),boxes(pick(i),4)-boxes(pick(i),2)],'EdgeColor','r','LineWidth',2);
end
axis ij;
axis equal;
axis([0 600 0 600]);
end

function pick = NMS(boxes, overlap)

% pick = nms(boxes, overlap) 
% Non-maximum suppression.
% Greedily select high-scoring detections and skip detections
% that are significantly covered by a previously selected detection.

if isempty(boxes)
  pick = [];
else
  x1 = boxes(:,1);          %���к�ѡ������ϽǶ���x 
  y1 = boxes(:,2);          %���к�ѡ������ϽǶ���y 
  x2 = boxes(:,3);          %���к�ѡ������½Ƕ���x 
  y2 = boxes(:,4);          %���к�ѡ������½Ƕ���y
  s = boxes(:,end);         %���к�ѡ������Ŷȣ����԰���1�л��߶��У����ڱ�ʾ��ͬ׼������Ŷ�
  area = (x2-x1+1) .* (y2-y1+1);%���к�ѡ������

  [vals, I] = sort(s);      %�����к�ѡ����д�С��������valsΪ���������IΪ������ǩ
  pick = [];
  while ~isempty(I)
    last = length(I);       %last�����ǩI�ĳ��ȣ������һ��Ԫ�ص�λ�ã���matlab�����1��ʼ������
    i = I(last);            %���к�ѡ��������Ŷ���ߵ��Ǹ��ı�ǩ��ֵ��i
    pick = [pick; i];       %��i����pick�У�pickΪһ�������������������NMS������box�����
    suppress = [last];      %��I��������Ŷȵı�ǩ��I��λ�ø�ֵ��suppress��suppress����Ϊ���ƴ��־��
                            %����suppress��֤����Ԫ�ش����
    for pos = 1:last-1      %��1�������ڶ�������ѭ��
      j = I(pos);           %�õ�posλ�õı�ǩ����ֵ��j
      xx1 = max(x1(i), x1(j));%���Ͻ�����x������������Ĺ�������
      yy1 = max(y1(i), y1(j));%���Ͻ�����y
      xx2 = min(x2(i), x2(j));%���½���С��x
      yy2 = min(y2(i), y2(j));%���½���С��y
      w = xx2-xx1+1;          %��������Ŀ��
      h = yy2-yy1+1;          %��������ĸ߶�
      if w > 0 && h > 0     %w,hȫ��>0��֤��2����ѡ���ཻ
        o = w * h / area(j);%����overlap��ֵ��������ռ��ѡ��j���������
        if o > overlap      %����������õ���ֵ��ȥ����ѡ��j����Ϊ��ѡ��i�����Ŷ����
          suppress = [suppress; pos];%���ڹ涨��ֵ�ͼ��뵽suppress��֤����Ԫ�ر������
        end
      end
    end
    I(suppress) = [];%���������suppress��Ϊ�գ���IΪ�ս���ѭ��
  end  
end
end
