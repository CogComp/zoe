var demoname = "Zoe Demo";
var demoexplanation = "This is an online demo of our recent paper Zero-Shot Open Entity Typing as Type-Compatible Grounding. Please use the question buttons when you are looking for instructions. If none of them solves your problem, please create an issue on our Github repo.";
var citations = {
	"http://cogcomp.org/page/publication_view/845" : "Zero-Shot Open Entity Typing as Type-Compatible Grounding",
};
var contact = "xzhou45@illinois.edu";

function initial_load() {
	document.getElementById("demo-name").innerHTML = demoname;
	document.getElementById("demo-explanation").innerHTML = "<p>" + demoexplanation + "</p>";
	if (citations.length != 0) {
		citation_content = "If you wish to cite this work, please cite the following publication(s):";
		var cid = 1;
		for (var key in citations) {
			citation_content +=
				"<span ng-repeat='pub in data.publications' class='ng-scope'>" +
                "<i class='ng-binding'>(" + cid.toString() + ")" +
                "<a href='" + key + "' class='ng-binding'>" + citations[key] + "</a></i><span ng-if='!$last' class='ng-scope'>,</span>" +
              	"</span>";
            cid ++;
		}
		document.getElementById("demo-citations").innerHTML = citation_content;
		document.getElementById("demo-contact").href = "mailto:" + contact;
		document.getElementById("demo-contact").innerHTML = contact;
	}
}

initial_load();