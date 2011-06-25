clear;
ntimes = 30;

fp = fopen('result.html','w');
fprintf(fp,'<html>\n<body>\nUsing Mappings : f(x) = x, f(x) = x.^2, f(x) = x.^3 ... f(x) = x.^30 as base mapping functions<br/>\n<table>');

fprintf(fp,'<tr>');
fprintf(fp,'<td></td>');
fprintf(fp,'<td>%d times accuracy mean +- stdev of liblinear linear feature</td>',ntimes);
fprintf(fp,'<td>%d times accuracy mean +- stdev of liblinear MKL</td>',ntimes);
fprintf(fp,'<td>%d times accuracy mean +- stdev of direct feature combination</td>',ntimes);
fprintf(fp,'</tr>\n');
for data={'../satimage.scale','../dna.scale','../glass.scale','../heart_scale','../ionosphere_scale','../diabetes_scale','../breast-cancer_scale','../sonar_scale'}
	clear newx
	clear func
	data = char(data);
	[y x] = libsvmread(data);
	[l n] = size(x);
	for j=1:30
		newx(:,(j-1)*n+1:j*n) = x.^j;
		func{j} = inline(sprintf('x.^%d',j));
	end

	sumacc = 0;
	sumacc_orig = 0;
	for i=1:ntimes
		subidx = randsample(l,ceil(l*4/5));
		predictidx = setdiff(1:l,subidx);
		yt = y(subidx);
		xt = x(subidx,:);
		[l n] = size(x);
	
		newx = sparse(newx);

		model = train(ones(size(x,2),1),yt,x(subidx,:),'-s 3');
		[py acc devs] = predict(y(predictidx,:),x(predictidx,:),model);
		linear_acc(i) = acc;

		model = train(ones(size(newx,2),1),yt,newx(subidx,:),'-s 3');
		[py acc devs] = predict(y(predictidx,:),newx(predictidx,:),model);
		orig_acc(i) = acc;

		mklmodel = train_mkl(yt,xt,func,'-s 3');
		[py acc devs] = predict_mkl(y(predictidx,:),x(predictidx,:),mklmodel);
		mkl_acc(i) = acc;
	end

%	disp(sprintf('\nResult of data set %s\n',data))
%	disp('Mean accuracy via selecting linear combination of mappings phi(x) = x and phi(x) = x.^2 (Primal MKL)');
%	acc = sumacc / 5
%	disp('Mean accuracy via concatenating features genereated by mappings phi(x) = x and phi(x) = x.^2');
%	acc_orig = sumacc_orig / 5

	fprintf(fp,'<tr>');
	fprintf(fp,'<td>%s</td>',data);
	fprintf(fp,'<td>%.4f +- %.4f</td>',mean(linear_acc),std(linear_acc));
	fprintf(fp,'<td>%.4f +- %.4f</td>',mean(mkl_acc),std(mkl_acc));
	fprintf(fp,'<td>%.4f +- %.4f</td>',mean(orig_acc),std(orig_acc));
	fprintf(fp,'</tr>\n');
end
fprintf(fp,'</table>\n</body>\n</html>');
fclose(fp);
