function [mklmodel] = train_mkl(y,x,phis,liblinear_param)
m = length(phis);

for i=1:m
	phi = phis{i};
	xadd = phi(x);
	if i == 1
		fstart(i) = 1;
		fend(i) = size(xadd,2);
	else
		fstart(i) = fend(i-1) + 1;
		fend(i) = fend(i-1) + size(xadd,2);
	end
	xs(:,fstart(i):fend(i)) = xadd;
end

nclass = length(unique(y));
classes = sort(unique(y));

if nclass == 2
	d = ones(1,m) ./ m;
	iter = 1;
	while true && iter < 100
		iter = iter + 1;
		ds = [];
		for i=1:m
			ds(fstart(i):fend(i)) = d(i);
		end
		model = train(ds',y,xs,sprintf('%s',liblinear_param));
		w = model.w;

		for i=1:m
			subw = w(fstart(i):fend(i));
			fh(i) = subw*subw';
		end
		for i=1:m
			newd(i) = fh(i) / sum(fh);
		end

		if norm(newd - d,2) < 1e-3
			d = newd;
			break;
		end
		d = newd;
	end
	mklmodel.model = model;
	mklmodel.d = d;
	mklmodel.classes = [];
elseif nclass > 2
	yc = zeros(length(y),1);
	for j = 1:nclass
		d = ones(1,m) ./ m;
		iter = 1;
		yc(find(y == classes(j))) = 1;
		yc(find(y ~= classes(j))) = -1;
		while true && iter < 100
			iter = iter + 1;
			ds = [];
			for i=1:m
				ds(fstart(i):fend(i)) = d(i);
			end
			model = train(ds',yc,xs,sprintf('%s',liblinear_param));
			w = model.w;

			for i=1:m
				subw = w(fstart(i):fend(i));
				fh(i) = subw*subw';
			end
			for i=1:m
				newd(i) = fh(i) / sum(fh);
			end

			if norm(newd - d,2) < 1e-3
				d = newd;
				break;
			end
			d = newd;
		end
		mklmodel.model{j} = model;
		mklmodel.d{j} = d;
	end
	mklmodel.classes = classes;
else
	disp('Error');
	mklmodel = [];
end
mklmodel.phis = phis;
