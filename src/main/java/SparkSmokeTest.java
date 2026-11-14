
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaSparkContext;

public class SparkSmokeTest {

    public static void main(String[] args) {

        SparkConf conf = new SparkConf()
                .setAppName("HealthcareIR")
                .setMaster("local[*]");

        try (JavaSparkContext sc = new JavaSparkContext(conf)) {

            long totalWords = sc.parallelize(
                    java.util.Arrays.asList("patient", "care", "patient")
            ).count();

            long distinctWords = sc.parallelize(
                    java.util.Arrays.asList("patient", "care", "patient")
            ).distinct().count();

            System.out.println("Total words = " + totalWords);
            System.out.println("Distinct words = " + distinctWords);
        }
    }
}
